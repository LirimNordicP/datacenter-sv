# Sweden/src/app.py

from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Dict, Optional, Tuple

import pandas as pd
import streamlit as st

from io_dropbox import list_parquet_files, read_parquet


# -----------------------------------------------------------------------------
# Page config & styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Nordic Pill Sweden Drug Data Center",
    page_icon="💊",
    layout="wide",
)

# Custom CSS
st.markdown(
    """
<style>
body {
    background: #030712;
    color: #e5e7eb;
}

.main-title {
    text-align: center;
    margin-bottom: 2rem;
}

.source-card {
    background: radial-gradient(circle at top left, #22d3ee33, #1f2937 55%, #020617);
    border-radius: 1.5rem;
    padding: 1.5rem 1.75rem;
    border: 1px solid rgba(148, 163, 184, 0.6);
    box-shadow: 0 18px 40px rgba(15, 23, 42, 0.9);
    height: 100%;
}

.source-title {
    font-size: 1.2rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
}

.source-subtitle {
    font-size: 0.9rem;
    color: #9ca3af;
    margin-bottom: 0.8rem;
}

.source-badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 999px;
    font-size: 0.75rem;
    border: 1px solid rgba(148, 163, 184, 0.7);
    color: #e5e7eb;
    margin-bottom: 0.6rem;
}

.source-meta {
    font-size: 0.85rem;
    margin-top: 0.6rem;
}

.stButton>button {
    border-radius: 999px !important;
    padding: 0.4rem 1.2rem !important;
    border: 1px solid rgba(148, 163, 184, 0.4) !important;
    background: linear-gradient(135deg, #0f172a, #020617) !important;
    color: #e5e7eb !important;
    font-weight: 600 !important;
    cursor: pointer !important;
}

.stButton>button:hover {
    border-color: #22d3ee !important;
    box-shadow: 0 0 0 1px rgba(56, 189, 248, 0.7) !important;
}

.block-container {
    padding-top: 3.5rem;
}
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Data source definitions
# -----------------------------------------------------------------------------

SOURCES: Dict[str, Dict] = {
    "lmv": {
        "label": "Läkemedelsverket",
        "site": "https://www.lakemedelsverket.se",
        "emoji": "🏛️",
        "description": "Nationellt produktregister för läkemedel (NPL).",
        "tables": {
            "npl_master": {
                "label": "NPL Master",
                "folder": "Sweden/npl_fetcher",
                "filename_prefix": "npl_master",
            }
        },
    },
    "soc": {
        "label": "Socialstyrelsen",
        "site": "https://www.socialstyrelsen.se",
        "emoji": "📊",
        "description": "Läkemedelsstatistik från Socialstyrelsen.",
        "tables": {
            "lakemedel": {
                "label": "Läkemedelsdata",
                "folder": "Sweden/soc_fetcher",
                "filename_prefix": "lakemedel",
            }
        },
    },
    "ehm": {
        "label": "E-hälsomyndigheten",
        "site": "https://www.ehalsomyndigheten.se",
        "emoji": "🌐",
        "description": "Vara-register: utbytbarhet, produkter, priser m.m.",
        "tables": {
            "INTERCHANGEABLE_VIEW": {
                "label": "Interchangeable View",
                "folder": "Sweden/vara_fetcher/INTERCHANGEABLE_VIEW",
                "filename_prefix": "vara_",
            },
            "PHARMACEUTICAL_VIEW": {
                "label": "Pharmaceutical View",
                "folder": "Sweden/vara_fetcher/PHARMACEUTICAL_VIEW",
                "filename_prefix": "vara_",
            },
            "PRICE_VIEW": {
                "label": "Price View",
                "folder": "Sweden/vara_fetcher/PRICE_VIEW",
                "filename_prefix": "vara_",
            },
            "PRODUCT_VIEW": {
                "label": "Product View",
                "folder": "Sweden/vara_fetcher/PRODUCT_VIEW",
                "filename_prefix": "vara_",
            },
            "TLV_SUBSTITUTION_VIEW": {
                "label": "TLV Substitution View",
                "folder": "Sweden/vara_fetcher/TLV_SUBSTITUTION_VIEW",
                "filename_prefix": "vara_",
            },
        },
    },
}


# -----------------------------------------------------------------------------
# Cached helper functions (critical for performance)
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_soc_years(parquet_path: str):
    """Load unique År values."""
    df = read_parquet(parquet_path, columns=["År"])
    return sorted(df["År"].dropna().unique())


@st.cache_data(show_spinner=False)
def load_soc_matt(parquet_path: str, year: int):
    """Load unique Mått values for a given År."""
    df = read_parquet(
        parquet_path,
        columns=["Mått"],
        filters=[("År", "=", year)],
    )
    return sorted(df["Mått"].dropna().unique())


@st.cache_data(show_spinner=True)
def load_soc_filtered(parquet_path: str, year: int, matt: str) -> pd.DataFrame:
    """Load main Socialstyrelsen dataset (År + Mått). One-time cached."""
    return read_parquet(
        parquet_path,
        filters=[("År", "=", year), ("Mått", "=", matt)],
        columns=None,
    )


@st.cache_data(show_spinner=True, ttl=24 * 60 * 60)
def load_parquet_from_dropbox(dropbox_path: str) -> pd.DataFrame:
    """Cached small/medium parquet loader."""
    return read_parquet(dropbox_path)


@st.cache_data(show_spinner=True, ttl=24 * 60 * 60)
def get_latest_parquet_in_folder(folder: str, prefix: Optional[str]):
    """Find latest parquet file based on YYYY-MM-DD in filename."""
    paths = list_parquet_files(folder)
    best = None

    for p in paths:
        fname = os.path.basename(p)
        if prefix and not fname.lower().startswith(prefix.lower()):
            continue

        m = re.search(r"(\d{4}-\d{2}-\d{2})", fname)
        if not m:
            continue

        dt = datetime.strptime(m.group(1), "%Y-%m-%d").date()
        best = max(best, dt) if best else dt

    if not best:
        return None, None

    for p in paths:
        if best.isoformat() in p:
            return p, best.isoformat()

    return None, None


def find_overall_latest_for_source(source_key: str):
    """Latest date across all tables for front-page cards."""
    best = None
    for tbl in SOURCES[source_key]["tables"].values():
        _, d = get_latest_parquet_in_folder(tbl["folder"], tbl.get("filename_prefix"))
        if d:
            dt = datetime.strptime(d, "%Y-%m-%d").date()
            if not best or dt > best:
                best = dt
    return best.isoformat() if best else None


# -----------------------------------------------------------------------------
# Filtering UI helper
# -----------------------------------------------------------------------------

def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    st.markdown("#### 🔍 Filter data")

    if df.empty:
        st.info("No data available.")
        return df

    df_filtered = df.copy()
    cols = st.multiselect("Choose columns to filter", df.columns)

    for col in cols:
        col_data = df_filtered[col]

        if pd.api.types.is_numeric_dtype(col_data):
            min_val, max_val = float(col_data.min()), float(col_data.max())
            selected = st.slider(
                f"{col} range", min_val, max_val, (min_val, max_val)
            )
            df_filtered = df_filtered[col_data.between(*selected)]

        elif pd.api.types.is_datetime64_any_dtype(col_data):
            min_date, max_date = col_data.min().date(), col_data.max().date()
            selected = st.date_input(f"{col} date range", (min_date, max_date))
            if len(selected) == 2:
                s, e = pd.to_datetime(selected[0]), pd.to_datetime(selected[1])
                df_filtered = df_filtered[col_data.between(s, e)]

        else:
            unique = col_data.dropna().unique()
            if len(unique) <= 200:
                vals = st.multiselect(f"{col} values", sorted(unique))
                if vals:
                    df_filtered = df_filtered[col_data.isin(vals)]
            else:
                pattern = st.text_input(f"Substring filter for {col}")
                if pattern:
                    df_filtered = df_filtered[
                        col_data.astype(str).str.contains(pattern, case=False, na=False)
                    ]

    return df_filtered


# -----------------------------------------------------------------------------
# Join helper
# -----------------------------------------------------------------------------

def join_with_uploaded(df: pd.DataFrame):
    st.markdown("### 🔗 Join with uploaded data")

    up = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"])
    if not up:
        st.info("Upload a file to join.")
        return

    try:
        df_up = pd.read_csv(up) if up.name.endswith(".csv") else pd.read_excel(up)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        return

    common_cols = sorted(df.columns.intersection(df_up.columns))

    if not common_cols:
        st.warning("No common columns.")
        return

    key = st.selectbox("Join key", common_cols)

    if st.button("Run left join"):
        joined = df.merge(df_up, on=key, how="left")
        st.success(f"Join complete: {joined.shape}")
        st.dataframe(joined)

        st.download_button(
            "Download CSV",
            joined.to_csv(index=False).encode(),
            "joined.csv",
            "text/csv",
        )


# -----------------------------------------------------------------------------
# Folder cards (home page)
# -----------------------------------------------------------------------------

def render_folder_cards():
    st.markdown(
        """
<div class="main-title">
  <h1>💊 Sweden Drug Data Center</h1>
  <p style="color:#9ca3af;">Utforska läkemedelsdata från tre statliga myndigheter.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    cols = st.columns(3)
    for col, (key, src) in zip(cols, SOURCES.items()):
        latest = find_overall_latest_for_source(key)
        with col:
            st.markdown(
                f"""
<div class="source-card">
  <div class="source-badge">{src["emoji"]} {src["label"]}</div>
  <div class="source-title">{src['site'].replace("https://","")}</div>
  <div class="source-subtitle">{src['description']}</div>
  <div class="source-meta">
    <strong>Latest update:</strong><br>{latest or "No files"}
  </div>
</div>
""",
                unsafe_allow_html=True,
            )
            if st.button(f"Open {src['label']}", key=f"open_{key}"):
                st.session_state["selected_source"] = key
                st.session_state.pop("selected_table_key", None)
                st.rerun()


# -----------------------------------------------------------------------------
# Source page
# -----------------------------------------------------------------------------

def render_source_page(source_key: str):
    source = SOURCES[source_key]

    if st.button("⬅️ Back to all sources"):
        st.session_state["selected_source"] = None
        st.session_state["selected_table_key"] = None
        st.rerun()

    st.markdown(
        f"## {source['emoji']} {source['label']} — {source['site'].replace('https://','')}"
    )
    st.markdown(source["description"])

    # -----------------------------------------------------
    # TABLE SELECTION
    # -----------------------------------------------------
    table_map = {}
    option_labels = []

    for tbl_key, tbl in source["tables"].items():
        path, date = get_latest_parquet_in_folder(
            tbl["folder"], tbl.get("filename_prefix")
        )
        label = tbl["label"]
        option_labels.append(f"{label} (latest: {date or 'none'})")
        table_map[label] = (path, date, tbl["folder"])

    selected_label = st.selectbox("Choose table", option_labels)
    table_label = selected_label.split(" (")[0]
    parquet_path, date_str, _folder = table_map[table_label]

    st.markdown(f"#### Table: **{table_label}** — Latest: {date_str}")

    if not parquet_path:
        st.error("No parquet file found.")
        return

    # -----------------------------------------------------
    # LOAD DATA
    # -----------------------------------------------------
    if source_key == "soc":
        st.markdown("### 🗃️ Select År and Mått")

        years = load_soc_years(parquet_path)
        chosen_year = st.selectbox("Choose År", years)

        matts = load_soc_matt(parquet_path, chosen_year)
        chosen_matt = st.selectbox("Choose Mått", matts)

        df_full = load_soc_filtered(parquet_path, chosen_year, chosen_matt)

        # Limit preview so browser never receives millions of rows
        MAX_PREVIEW = 200_000
        df_preview_base = df_full.head(MAX_PREVIEW)

    else:
        df_full = load_parquet_from_dropbox(parquet_path)
        df_preview_base = df_full  # no preview limit for smaller datasets

    st.caption(f"Loaded (preview) {df_preview_base.shape[0]} rows × {df_preview_base.shape[1]} columns")

    # -----------------------------------------------------
    # COLUMN SELECTION (SILENTLY CONTROLS AGGREGATION FOR SOC)
    # -----------------------------------------------------
    all_cols = list(df_preview_base.columns)
    default_cols = all_cols[: min(25, len(all_cols))]
    selected_cols = st.multiselect("Select columns", all_cols, default_cols)
    if not selected_cols:
        selected_cols = all_cols

    # -----------------------------------------------------
    # SOCIALSTYRELSEN: AUTOMATIC AGGREGATION (no extra preview)
    # -----------------------------------------------------
    if source_key == "soc":
        group_cols = [c for c in selected_cols if c != "Värde"]

        if not group_cols:
            # Collapse everything into one row
            df_full_agg = (
                df_full.groupby([])["Värde"]
                .sum()
                .reset_index()
                .rename(columns={"Värde": "Värde_sum"})
            )
        else:
            df_full_agg = (
                df_full.groupby(group_cols)["Värde"]
                .sum()
                .reset_index()
                .rename(columns={"Värde": "Värde_sum"})
            )

        # Apply preview to aggregated dataset
        MAX_PREVIEW = 200_000
        df_preview = df_full_agg.head(MAX_PREVIEW)

        st.caption(
            f"Full aggregated shape: {df_full_agg.shape[0]:,} rows × {df_full_agg.shape[1]} columns "
            f"• Preview: {df_preview.shape[0]:,} rows"
        )

    else:
        # For other sources, just show selected columns
        df_preview = df_preview_base[selected_cols]

    # -----------------------------------------------------
    # FILTER UI (works on preview)
    # -----------------------------------------------------
    df_final = filter_dataframe(df_preview)

    # -----------------------------------------------------
    # FINAL PREVIEW (only ONE preview section!)
    # -----------------------------------------------------
    st.markdown("### 📋 Data Preview")
    st.dataframe(df_final, use_container_width=True)
    st.caption(f"Preview shape: {df_final.shape[0]} rows × {df_final.shape[1]} columns")

    # -----------------------------------------------------
    # JOIN
    # -----------------------------------------------------
    join_with_uploaded(df_final)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    if "selected_source" not in st.session_state:
        st.session_state["selected_source"] = None

    if st.session_state["selected_source"] is None:
        render_folder_cards()
    else:
        render_source_page(st.session_state["selected_source"])


if __name__ == "__main__":
    main()
