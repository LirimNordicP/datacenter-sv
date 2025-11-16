# Sweden/src/io_dropbox.py
from __future__ import annotations

import io
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd  # you'll load into a placeholder 'df' in your app
import dropbox
from dropbox.exceptions import AuthError, ApiError

from settings import settings

# ---- logging (works even if you don't have logging_utils yet) ----
try:
    from logging_utils import get_logger

    log = get_logger("io_dropbox")
except Exception:  # pragma: no cover
    import logging

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    log = logging.getLogger("io_dropbox")


# --------------------------
# Internal path helpers
# --------------------------
def _norm_folder(folder_path: str | None) -> str:
    """
    Normalize a Dropbox App Folder subpath.
    '' or '/' → app root. Ensures leading '/' for non-empty paths and strips trailing slash.
    """
    if not folder_path or folder_path.strip() in {"", "/"}:
        return ""
    fp = folder_path.strip()
    if not fp.startswith("/"):
        fp = "/" + fp
    if len(fp) > 1 and fp.endswith("/"):
        fp = fp[:-1]
    return fp


def _ensure_leading_slash(path: str) -> str:
    path = path.replace("\\", "/")
    return path if path.startswith("/") else f"/{path}"


# --------------------------
# Dropbox auth / client
# --------------------------

_DBX: Optional[dropbox.Dropbox] = None
_DBX_AUTHED: bool = False


def _build_client() -> dropbox.Dropbox:
    """
    Prefer refresh-token auth; fallback to legacy token if provided.
    Reuse a module-level client to avoid repeated auth churn/logging.
    """
    global _DBX, _DBX_AUTHED
    if _DBX is not None:
        return _DBX

    app_key = settings.DROPBOX_APP_KEY
    refresh = (
        settings.DROPBOX_REFRESH_TOKEN.get_secret_value()
        if settings.DROPBOX_REFRESH_TOKEN
        else None
    )
    app_secret = (
        settings.DROPBOX_APP_SECRET.get_secret_value()
        if getattr(settings, "DROPBOX_APP_SECRET", None)
        else None
    )

    if refresh:
        _DBX = dropbox.Dropbox(
            oauth2_access_token=None,
            oauth2_refresh_token=refresh,
            app_key=app_key,
            app_secret=app_secret,
        )
    elif settings.DROPBOX_TOKEN is not None:
        _DBX = dropbox.Dropbox(settings.DROPBOX_TOKEN.get_secret_value())
    else:
        raise RuntimeError(
            "Dropbox credentials missing: set DROPBOX_APP_KEY and DROPBOX_REFRESH_TOKEN in .env"
        )

    # Authenticate only once for a friendly log line; skip on subsequent uses
    if not _DBX_AUTHED:
        try:
            acc = _DBX.users_get_current_account()
            log.info("Authenticated Dropbox as: %s", acc.name.display_name)
        except AuthError as e:
            raise RuntimeError(f"Dropbox authentication failed: {e}") from e
        _DBX_AUTHED = True

    return _DBX


# --------------------------
# Core listing/reading API
# --------------------------
def list_parquet_files(
    folder_path: str,
    *,
    recursive: bool = False,
    dbx: Optional[dropbox.Dropbox] = None,
) -> List[str]:
    """
    Return full Dropbox paths (e.g., '/Sweden/npl_fetcher/file.parquet')
    for all .parquet files under `folder_path`.
    """
    dbx = dbx or _build_client()
    base = _norm_folder(folder_path)

    try:
        result = dbx.files_list_folder(base, recursive=recursive)
    except ApiError as e:
        log.error("Dropbox list error for %s: %s", base or "/", e)
        return []

    paths: List[str] = []
    entries = list(result.entries)
    while result.has_more:
        result = dbx.files_list_folder_continue(result.cursor)
        entries.extend(result.entries)

    for entry in entries:
        if isinstance(
            entry, dropbox.files.FileMetadata
        ) and entry.name.lower().endswith(".parquet"):
            paths.append(entry.path_lower)  # lower-cased canonical path
    return paths


def read_parquet(
    dropbox_path: str,
    *,
    dbx: Optional[dropbox.Dropbox] = None,
    columns: Optional[Iterable[str]] = None,
    filters: Optional[list] = None,
) -> pd.DataFrame:
    """
    Read a parquet file from Dropbox using pyarrow.
    Supports predicate pushdown (filters) for massive speed gains.
    Example filter:
        filters=[("År", "=", 2022)]
    """
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    dbx = dbx or _build_client()
    dp = _ensure_leading_slash(dropbox_path)

    try:
        _, resp = dbx.files_download(dp)
    except ApiError as e:
        raise FileNotFoundError(f"Dropbox download failed for {dp}: {e}") from e

    buf = io.BytesIO(resp.content)

    # Use PyArrow dataset API for predicate filtering
    table = pq.read_table(
        buf,
        columns=list(columns) if columns else None,
        filters=filters,
    )

    return table.to_pandas()



def get_file_signature(
    dropbox_path: str, *, dbx: Optional[dropbox.Dropbox] = None
) -> str:
    """
    Returns a stable signature for cache invalidation.
    Prefer Dropbox file 'rev' (changes on every update). Fallback to size+timestamp.
    """
    dbx = dbx or _build_client()
    dp = _ensure_leading_slash(dropbox_path)
    md = dbx.files_get_metadata(dp)
    if isinstance(md, dropbox.files.FileMetadata):
        # rev is perfect for invalidation
        if getattr(md, "rev", None):
            return f"{md.id}:{md.rev}"
        # fallback when rev is not present for some reason
        return f"{md.path_lower}:{md.size}:{md.server_modified.isoformat()}"
    # Non-file? Use path only as a last resort
    return dp


def read_folder_parquets(
    folder_path: str,
    *,
    recursive: bool = False,
    limit: Optional[int] = None,
    columns: Optional[Iterable[str]] = None,
    concat: bool = True,
    dbx: Optional[dropbox.Dropbox] = None,
) -> pd.DataFrame | List[pd.DataFrame]:
    """
    Read all (or first N) parquet files under a folder.
    - If concat=True (default): returns a single concatenated df (empty if none).
    - If concat=False: returns a list of per-file dfs.
    """
    dbx = dbx or _build_client()
    paths = list_parquet_files(folder_path, recursive=recursive, dbx=dbx)
    if not paths:
        log.warning("No parquet files found in: %s", folder_path)
        return pd.DataFrame() if concat else []

    if limit is not None:
        paths = paths[: int(limit)]

    dfs: List[pd.DataFrame] = []
    for p in paths:
        df = read_parquet(p, dbx=dbx, columns=columns)
        dfs.append(df)

    if concat:
        # If no files (shouldn't happen here), this is an empty DataFrame
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    return dfs


def read_from_configured_folders(
    *,
    recursive: bool = False,
    limit_per_folder: Optional[int] = None,
    columns: Optional[Iterable[str]] = None,
    concat: bool = True,
) -> pd.DataFrame | List[pd.DataFrame]:
    """
    Convenience: iterate settings.dropbox_folders and read parquets.
    - If concat=True (default): concatenates all folders into a single df.
    - If concat=False: returns a list [df_folder1, df_folder2, ...] in same order.
    """
    dbx = _build_client()
    folder_list = settings.dropbox_folders  # already normalized in settings.py
    if not folder_list:
        log.warning("settings.dropbox_folders is empty. Using default inside settings.")
        folder_list = settings.dropbox_folders

    per_folder: List[pd.DataFrame] = []
    for f in folder_list:
        log.info("Reading folder: %s", f)
        df_folder = read_folder_parquets(
            f,
            recursive=recursive,
            limit=limit_per_folder,
            columns=columns,
            concat=True,  # concat within the folder
            dbx=dbx,
        )
        # ensure we always have a DataFrame to stack
        if isinstance(df_folder, list):
            df_folder = (
                pd.concat(df_folder, ignore_index=True) if df_folder else pd.DataFrame()
            )
        per_folder.append(df_folder)

    if concat:
        return (
            pd.concat(per_folder, ignore_index=True) if per_folder else pd.DataFrame()
        )
    return per_folder
