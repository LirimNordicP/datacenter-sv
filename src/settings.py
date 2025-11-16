# Sweden/src/settings.py

from __future__ import annotations
from pathlib import Path
from typing import List
import json
from pydantic import SecretStr, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _split_to_list(v: str | List[str] | None) -> List[str]:
    """
    Accepts a string like:
      "Sweden/npl_fetcher, Sweden/soc_fetcher"
      "Sweden/npl_fetcher;Sweden/soc_fetcher"
      "Sweden/npl_fetcher\nSweden/soc_fetcher"
    …and returns a clean list. If already a list, returns as-is.
    """
    if v is None:
        return []
    if isinstance(v, list):
        return [x.strip() for x in v if x and str(x).strip()]
    # split on comma, semicolon, or newline
    parts = []
    for sep in [",", ";", "\n"]:
        if sep in v:
            parts = [p.strip() for p in v.split(sep)]
            break
    if not parts:
        parts = [v.strip()]
    return [p for p in parts if p]


class Settings(BaseSettings):
    """
    Runtime configuration loaded from environment / .env.
    Tailored for a Streamlit app that reads Parquet files from Dropbox.
    """

    # --- App / Logging ---
    APP_NAME: str = "datacenter"
    LOG_LEVEL: str = "INFO"

    # --- Local data cache (optional) ---
    # Where downloaded/temporary Parquet files can be cached locally if you choose to.
    DATA_DIR: str = "./data/parquet_cache"

    # --- Dropbox OAuth (use refresh-token flow if possible) ---
    DROPBOX_APP_KEY: str = Field(default="", description="Dropbox App Key")
    DROPBOX_APP_SECRET: SecretStr | None = None  # optional if PKCE used
    DROPBOX_REFRESH_TOKEN: SecretStr | None = Field(
        default=None, description="Dropbox OAuth2 refresh token (offline)"
    )

    # --- Dropbox short-lived access token (legacy/optional) ---
    DROPBOX_TOKEN: SecretStr | None = None

    # --- Dropbox folders to read Parquet from ---
    # Preferred: set DROPBOX_FOLDERS in .env (comma/semicolon/newline separated)
    # Backwards compat: if only DROPBOX_FOLDER is set, we use that.
    DROPBOX_FOLDERS: str | None = None
    DROPBOX_FOLDER: str | None = None  # single-folder fallback

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parents[1] / ".env"),  # Sweden/.env
        case_sensitive=False,
        extra="ignore",
    )

    # ---- Convenience properties ----
    @property
    def data_path(self) -> Path:
        return Path(self.DATA_DIR).resolve()

    @property
    def dropbox_folders(self) -> List[str]:
        raw = self.DROPBOX_FOLDERS
        if raw:
            # Allow JSON (["Sweden/npl_fetcher", "Norway/soc_fetcher"]) OR delimited strings
            try:
                if raw.strip().startswith("["):
                    parsed = json.loads(raw)
                    return _split_to_list(parsed)
            except Exception:
                pass
            return _split_to_list(raw)

        if self.DROPBOX_FOLDER and self.DROPBOX_FOLDER.strip():
            return [self.DROPBOX_FOLDER.strip()]

        return ["Sweden/npl_fetcher"]


# Singleton
settings = Settings()
