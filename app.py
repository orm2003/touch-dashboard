import os
import gc
import ssl
import ast
import hashlib
import urllib.request
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

# =============================================================================
# Streamlit page + debug
# =============================================================================
st.set_page_config(
    page_title="Touch Recommendation Dashboard",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)
# Show full exceptions in the app (useful on Cloud)
st.set_option("client.showErrorDetails", True)

# Make pandas less copy-happy (saves memory on filters/transforms)
pd.options.mode.copy_on_write = True

# =============================================================================
# Constants
# =============================================================================
MONTH_NAMES = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December"
}
MONTH_ABBR = {m: MONTH_NAMES[m][:3] for m in MONTH_NAMES}

PASSWORD = "msba@touch"

# Google Drive parquet file (≈35MB)
GDRIVE_FILE_ID = "1cPFjQmRzuZhwq0Q1S4l8fihhWwHhIcQs"
DATA_FILE_NAME = "data.parquet"

# =============================================================================
# Offer catalog (small; negligible memory)
# =============================================================================
OFFER_CATALOG = {
    "Legacy HS Series": pd.DataFrame([
        {"Code": "HS1 POST", "Name": "HS1 Post", "Type": "Postpaid", "Data (MB)": 500, "Price ($)": 3.5, "Voice": 0},
        {"Code": "HS2 POST", "Name": "HS2 Post", "Type": "Postpaid", "Data (MB)": 1792, "Price ($)": 6, "Voice": 0},
        {"Code": "HS3 POST", "Name": "HS3 Post", "Type": "Postpaid", "Data (MB)": 6144, "Price ($)": 8.5, "Voice": 0},
        {"Code": "HS4 POST", "Name": "HS4 Post", "Type": "Postpaid", "Data (MB)": 10240, "Price ($)": 11, "Voice": 0},
        {"Code": "HS5 POST", "Name": "HS5 Post", "Type": "Postpaid", "Data (MB)": 20480, "Price ($)": 13, "Voice": 0},
        {"Code": "HS6 POST", "Name": "HS6 Post", "Type": "Postpaid", "Data (MB)": 30720, "Price ($)": 16, "Voice": 0},
        {"Code": "HS7 POST", "Name": "HS7 Post", "Type": "Postpaid", "Data (MB)": 40960, "Price ($)": 19.5, "Voice": 0},
        {"Code": "HS8 POST", "Name": "HS8 Post", "Type": "Postpaid", "Data (MB)": 61440, "Price ($)": 23, "Voice": 0},
        {"Code": "HS9 POST", "Name": "HS9 Post", "Type": "Postpaid", "Data (MB)": 102400, "Price ($)": 36, "Voice": 0},
        {"Code": "HS10 POST", "Name": "HS10 Post", "Type": "Postpaid", "Data (MB)": 204800, "Price ($)": 66, "Voice": 0},
        {"Code": "HS11 POST", "Name": "HS11 Post", "Type": "Postpaid", "Data (MB)": 409600, "Price ($)": 116, "Voice": 0},
        {"Code": "HS1 PREP", "Name": "HS1 Prepaid", "Type": "Prepaid", "Data (MB)": 500, "Price ($)": 3.5, "Voice": 0},
        {"Code": "HS2 PREP", "Name": "HS2 Prepaid", "Type": "Prepaid", "Data (MB)": 1792, "Price ($)": 6, "Voice": 0},
        {"Code": "HS3 PREP", "Name": "HS3 Prepaid", "Type": "Prepaid", "Data (MB)": 6144, "Price ($)": 8.5, "Voice": 0},
        {"Code": "HS4 PREP", "Name": "HS4 Prepaid", "Type": "Prepaid", "Data (MB)": 10240, "Price ($)": 11, "Voice": 0},
        {"Code": "HS5 PREP", "Name": "HS5 Prepaid", "Type": "Prepaid", "Data (MB)": 20480, "Price ($)": 13, "Voice": 0},
        {"Code": "HS6 PREP", "Name": "HS6 Prepaid", "Type": "Prepaid", "Data (MB)": 30720, "Price ($)": 16, "Voice": 0},
        {"Code": "HS7 PREP", "Name": "HS7 Prepaid", "Type": "Prepaid", "Data (MB)": 40960, "Price ($)": 19.5, "Voice": 0},
        {"Code": "HS8 PREP", "Name": "HS8 Prepaid", "Type": "Prepaid", "Data (MB)": 61440, "Price ($)": 23, "Voice": 0},
        {"Code": "HS9 PREP", "Name": "HS9 Prepaid", "Type": "Prepaid", "Data (MB)": 102400, "Price ($)": 36, "Voice": 0},
        {"Code": "HS10 PREP", "Name": "HS10 Prepaid", "Type": "Prepaid", "Data (MB)": 204800, "Price ($)": 66, "Voice": 0},
        {"Code": "HS11 PREP", "Name": "HS11 Prepaid", "Type": "Prepaid", "Data (MB)": 409600, "Price ($)": 116, "Voice": 0},
    ]),
    "M Series (HS Replacement)": pd.DataFrame([
        {"Code": "M1 POST", "Name": "M1 Post", "Type": "Postpaid", "Data (MB)": 1024, "Price ($)": 3.5, "Voice": 0},
        {"Code": "M7 POST", "Name": "M7 Post", "Type": "Postpaid", "Data (MB)": 7168, "Price ($)": 9, "Voice": 0},
        {"Code": "M22 POST", "Name": "M22 Post", "Type": "Postpaid", "Data (MB)": 22528, "Price ($)": 14.5, "Voice": 0},
        {"Code": "M44 POST", "Name": "M44 Post", "Type": "Postpaid", "Data (MB)": 45056, "Price ($)": 21, "Voice": 0},
        {"Code": "M77 POST", "Name": "M77 Post", "Type": "Postpaid", "Data (MB)": 78848, "Price ($)": 31, "Voice": 0},
        {"Code": "M111 POST", "Name": "M111 Post", "Type": "Postpaid", "Data (MB)": 113664, "Price ($)": 40, "Voice": 0},
        {"Code": "M444 POST", "Name": "M444 Post", "Type": "Postpaid", "Data (MB)": 454656, "Price ($)": 129, "Voice": 0},
        {"Code": "M1 PREP", "Name": "M1 Prepaid", "Type": "Prepaid", "Data (MB)": 1024, "Price ($)": 3.5, "Voice": 0},
        {"Code": "M7 PREP", "Name": "M7 Prepaid", "Type": "Prepaid", "Data (MB)": 7168, "Price ($)": 9, "Voice": 0},
        {"Code": "M22 PREP", "Name": "M22 Prepaid", "Type": "Prepaid", "Data (MB)": 22528, "Price ($)": 14.5, "Voice": 0},
        {"Code": "M44 PREP", "Name": "M44 Prepaid", "Type": "Prepaid", "Data (MB)": 45056, "Price ($)": 21, "Voice": 0},
        {"Code": "M77 PREP", "Name": "M77 Prepaid", "Type": "Prepaid", "Data (MB)": 78848, "Price ($)": 31, "Voice": 0},
        {"Code": "M111 PREP", "Name": "M111 Prepaid", "Type": "Prepaid", "Data (MB)": 113664, "Price ($)": 40, "Voice": 0},
        {"Code": "M444 PREP", "Name": "M444 Prepaid", "Type": "Prepaid", "Data (MB)": 454656, "Price ($)": 129, "Voice": 0},
    ]),
    "Web & Talk Bundles": pd.DataFrame([
        {"Code": "WEB AND TALK PREP", "Name": "Web & Talk Standard", "Type": "Prepaid", "Data (MB)": 600, "Price ($)": 4.67, "Voice": 60},
        {"Code": "WEB AND TALK MINI 1 PREP", "Name": "Web & Talk Mini 1", "Type": "Prepaid", "Data (MB)": 1024, "Price ($)": 4.9, "Voice": 30},
        {"Code": "WEB AND TALK MINI 2 PREP", "Name": "Web & Talk Mini 2", "Type": "Prepaid", "Data (MB)": 3072, "Price ($)": 6.9, "Voice": 60},
        {"Code": "WEB AND TALK MAXI 1 PREP", "Name": "Web & Talk Maxi 1", "Type": "Prepaid", "Data (MB)": 9216, "Price ($)": 9.9, "Voice": 90},
        {"Code": "WEB AND TALK MAXI 2 PREP", "Name": "Web & Talk Maxi 2", "Type": "Prepaid", "Data (MB)": 25600, "Price ($)": 14.9, "Voice": 120},
        {"Code": "WEB AND TALK MINI 1", "Name": "Web & Talk Mini 1", "Type": "Postpaid", "Data (MB)": 1024, "Price ($)": 4.4, "Voice": 100},
        {"Code": "WEB AND TALK MINI 2", "Name": "Web & Talk Mini 2", "Type": "Postpaid", "Data (MB)": 2048, "Price ($)": 7.7, "Voice": 200},
        {"Code": "WEB AND TALK LIGHT", "Name": "Web & Talk Light", "Type": "Postpaid", "Data (MB)": 3072, "Price ($)": 16.34, "Voice": 600},
        {"Code": "WEB AND TALK POST", "Name": "Web & Talk", "Type": "Postpaid", "Data (MB)": 6144, "Price ($)": 33, "Voice": 1200},
        {"Code": "WEB AND TALK ELITE", "Name": "Web & Talk Elite", "Type": "Postpaid", "Data (MB)": 30720, "Price ($)": 66.34, "Voice": 2400},
    ]),
    "Social & OTT Bundles": pd.DataFrame([
        {"Code": "WHATSAPP", "Name": "WhatsApp Bundle", "Type": "Prepaid", "Data (MB)": 200, "Price ($)": 1.34, "Voice": 0},
        {"Code": "WHATSAPP2", "Name": "WhatsApp Bundle 2", "Type": "Prepaid", "Data (MB)": 300, "Price ($)": 2, "Voice": 0},
        {"Code": "SOCIALBUNDLE", "Name": "Social Data Bundle", "Type": "Prepaid", "Data (MB)": 300, "Price ($)": 2.34, "Voice": 0},
        {"Code": "SOCIAL BUNDLE POST", "Name": "Social Bundle", "Type": "Postpaid", "Data (MB)": 1024, "Price ($)": 3, "Voice": 0},
    ]),
    "Special Bundles": pd.DataFrame([
        {"Code": "BILKHIDMEH BUNDLE", "Name": "Bil Khidmeh Bundle", "Type": "Prepaid", "Data (MB)": 1536, "Price ($)": 1.5, "Voice": 120},
        {"Code": "STUDENTBUNDLE", "Name": "Student Bundle", "Type": "Prepaid", "Data (MB)": 5120, "Price ($)": 5, "Voice": 60},
        {"Code": "TAWASOL BUNDLE", "Name": "Tawasol Bundle", "Type": "Prepaid", "Data (MB)": 100, "Price ($)": 4.5, "Voice": 10},
        {"Code": "MA3AKBUNDLE", "Name": "Ma3ak Bundle", "Type": "Prepaid", "Data (MB)": 1024, "Price ($)": 1.5, "Voice": 120},
        {"Code": "VISITORBUNDLE", "Name": "Visitor Bundle", "Type": "Prepaid", "Data (MB)": 10240, "Price ($)": 13, "Voice": 100},
    ]),
    "Limited-Time Promotions": pd.DataFrame([
        {"Code": "MOTHER'S DAY PROMO", "Name": "Mother's Day Promo", "Type": "Prepaid", "Data (MB)": 2048, "Price ($)": 1, "Voice": 60},
        {"Code": "LABOR DAY PROMO", "Name": "Labor Day Promo", "Type": "Prepaid", "Data (MB)": 1024, "Price ($)": 1, "Voice": 60},
        {"Code": "VALENTINE'S PROMO", "Name": "Valentine's Promo", "Type": "Prepaid", "Data (MB)": 2048, "Price ($)": 1.4, "Voice": 60},
        {"Code": "FATHER'S DAY PROMO", "Name": "Father's Day Promo", "Type": "Prepaid", "Data (MB)": 2048, "Price ($)": 1.5, "Voice": 60},
    ]),
    "Short-Term Data Bundles": pd.DataFrame([
        {"Code": "2 HOURS DATA BUNDLE", "Name": "HD2", "Type": "Prepaid", "Data (MB)": 20, "Price ($)": 0.1, "Voice": 0},
        {"Code": "6 HOURS DATA BUNDLE", "Name": "HD6", "Type": "Prepaid", "Data (MB)": 50, "Price ($)": 0.2, "Voice": 0},
        {"Code": "DDB", "Name": "Daily Data Bundle", "Type": "Prepaid", "Data (MB)": 50, "Price ($)": 0.34, "Voice": 0},
        {"Code": "2DDB", "Name": "2 Days Data Bundle", "Type": "Prepaid", "Data (MB)": 200, "Price ($)": 0.67, "Voice": 0},
        {"Code": "DD4", "Name": "4 Days Data Bundle", "Type": "Prepaid", "Data (MB)": 700, "Price ($)": 1.34, "Voice": 0},
        {"Code": "WDB", "Name": "Weekly Data Bundle", "Type": "Prepaid", "Data (MB)": 1024, "Price ($)": 2.34, "Voice": 0},
    ]),
    "Postpaid Snacks": pd.DataFrame([
        {"Code": "SNACK BUNDLE1", "Name": "Snack Bundle 1", "Type": "Postpaid", "Data (MB)": 100, "Price ($)": 1, "Voice": 0},
        {"Code": "SNACK BUNDLE2", "Name": "Snack Bundle 2", "Type": "Postpaid", "Data (MB)": 200, "Price ($)": 1.34, "Voice": 0},
        {"Code": "SNACK BUNDLE3", "Name": "Snack Bundle 3", "Type": "Postpaid", "Data (MB)": 300, "Price ($)": 1.67, "Voice": 0},
    ])
}

# =============================================================================
# Helpers
# =============================================================================
def _download_gdrive_file(dst_path: str) -> None:
    """Chunked download from Google Drive to local file (no Streamlit calls)."""
    if os.path.exists(dst_path):
        return
    url = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(url, context=ctx, timeout=60) as r, open(dst_path, "wb") as out:
        CHUNK = 1 << 14  # 16KB
        while True:
            b = r.read(CHUNK)
            if not b:
                break
            out.write(b)

def _normalize_offer_to_str(val: object) -> str:
    """Return a compact, display-friendly string. Avoid storing list objects (saves memory)."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "None"
    if isinstance(val, str):
        s = val.strip()
        if not s or s.upper() in {"NONE", "NULL", "NAN", "NO_RECOMMENDATION"}:
            return "None"
        # Try to parse list/tuple string
        if (s.startswith("(") and s.endswith(")")) or (s.startswith("[") and s.endswith("]")):
            try:
                lit = ast.literal_eval(s)
                if isinstance(lit, (list, tuple)):
                    parts = [str(x).strip() for x in lit if x and str(x).strip()]
                    return ", ".join(parts) if parts else "None"
                return str(lit).strip() or "None"
            except Exception:
                pass
        # Comma-separated fallback
        if "," in s:
            parts = [p.strip().strip("'\"") for p in s.split(",")]
            parts = [p for p in parts if p and p.upper() not in {"NONE", "NULL", "NAN"}]
            return ", ".join(parts) if parts else "None"
        return s
    if isinstance(val, (list, tuple)):
        parts = []
        for x in val:
            if x is None or (isinstance(x, float) and pd.isna(x)):
                continue
            s = str(x).strip()
            if s and s.upper() not in {"NONE", "NULL", "NAN"}:
                parts.append(s)
        return ", ".join(parts) if parts else "None"
    s = str(val).strip()
    return s if s else "None"

def _downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns to reduce memory footprint."""
    for col in df.select_dtypes(include=["integer", "Int64", "float"]).columns:
        if col == "MONTH":
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int16")
            continue
        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce", downcast="integer")
        elif pd.api.types.is_float_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce", downcast="float")
    return df

def _astype_categories(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        if c in df.columns and df[c].dtype == "object":
            df[c] = df[c].astype("category")

def _is_no_rec_string(s: object) -> bool:
    """Robust 'no recommendation' detector used in analytics stats."""
    if s is None:
        return True
    s = str(s).strip()
    if not s:
        return True
    return s.upper() in {"NONE", "NULL", "NAN", "NO_RECOMMENDATION"}

# =============================================================================
# Data loading (pure inside cache; NO st.* in here)
# =============================================================================
@st.cache_data(persist=True, show_spinner=False)
def load_data():
    """
    Load + preprocess data from Google Drive.
    Memory optimizations:
      - numeric downcasting
      - string->category for key columns (incl. CustomerID)
      - MONTH forced to int8 (1-12)
      - store only offer strings + boolean flag (no list columns)
    """
    try:
        _download_gdrive_file(DATA_FILE_NAME)
        df = pd.read_parquet(DATA_FILE_NAME)

        # Downcast numbers first
        df = _downcast_numeric(df)

        # MONTH cleanup -> int8
        if "MONTH" in df.columns:
            df["MONTH"] = pd.to_numeric(df["MONTH"], errors="coerce").fillna(0).astype("int16")
            df = df[df["MONTH"].between(1, 12)]
            df["MONTH"] = df["MONTH"].astype("int8")

        # Known numeric columns -> tighter floats/ints
        for c in ["ARPU", "Price_Difference", "Recommended_Offer_Price", "MB_CONSUMPTION",
                  "mb_allowance", "mb_usage_pct"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce", downcast="float")
        if "MINUTES" in df.columns:
            df["MINUTES"] = pd.to_numeric(df["MINUTES"], errors="coerce", downcast="integer")

        # String-like IDs as category (big win if many repeats)
        if "CustomerID" in df.columns:
            df["CustomerID"] = df["CustomerID"].astype(str).astype("category")

        # Key categoricals
        _astype_categories(df, [
            "Customer_Type", "Liquidity_Persona", "Consumption_Persona",
            "Sub_Persona", "Device_Category", "DEVICE_MODEL"
        ])

        # Offers: keep only compact string + boolean
        if "offer_pattern" in df.columns:
            df["offer_pattern_str"] = df["offer_pattern"].apply(_normalize_offer_to_str)
            df.drop(columns=["offer_pattern"], inplace=True)
        else:
            df["offer_pattern_str"] = "None"

        if "Recommended_Offer_Pattern" in df.columns:
            df["recommended_offers_str"] = df["Recommended_Offer_Pattern"].apply(_normalize_offer_to_str)
            df.drop(columns=["Recommended_Offer_Pattern"], inplace=True)
        else:
            df["recommended_offers_str"] = "None"

        # boolean: has recommendation
        df["recommended_has"] = ~df["recommended_offers_str"].apply(_is_no_rec_string)

        # Ensure Price_Difference exists and is float32
        if "Price_Difference" not in df.columns:
            df["Price_Difference"] = np.float32(0.0)
        else:
            df["Price_Difference"] = df["Price_Difference"].astype("float32")

        gc.collect()
        return df
    except Exception:
        return None

# =============================================================================
# Auth (simple)
# =============================================================================
def check_password() -> bool:
    """Returns True if password is correct; otherwise shows login UI."""
    def password_entered():
        if hashlib.sha256(st.session_state["password"].encode()).hexdigest() == hashlib.sha256(PASSWORD.encode()).hexdigest():
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("<br><br><br>", unsafe_allow_html=True)
            try:
                logo = Image.open("touch.png")
                st.image(logo, width=200, use_container_width=False)
            except Exception:
                st.markdown("## 📱 TOUCH")
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                "<div style='text-align: center; padding: 20px; background: #f0f2f6; border-radius: 10px;'>"
                "<h3 style='color: #1A73E8;'>Access Portal</h3>"
                "<p style='color: #5F6368;'>Please enter your password to continue</p>"
                "</div>",
                unsafe_allow_html=True,
            )
            st.text_input("Password", type="password", on_change=password_entered, key="password",
                          placeholder="Enter password...")
            st.markdown(
                "<div style='text-align: center; margin-top: 50px; color: #999;'>"
                "<small>Touch Recommendation Dashboard Using 2 way Clustering</small><br>"
                "<small>Touch MSBA Capstone Project | Omar Mneimne.</small>"
                "</div>",
                unsafe_allow_html=True,
            )
        return False
    elif not st.session_state["password_correct"]:
        col1, col2, col3 = st.columns([2, 2, 2])
        with col2:
            st.error("😕 Incorrect password. Please try again.")
            st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False
    else:
        return True

# =============================================================================
# Charts + metrics
# =============================================================================
def create_kpi_metrics(df: pd.DataFrame, selected_month: int | None = None):
    month_df = df[df["MONTH"] == selected_month] if selected_month else df
    metrics = {
        "total_customers": df["CustomerID"].nunique() if "CustomerID" in df.columns else 0,
        "total_arpu": df["ARPU"].sum() if "ARPU" in df.columns else 0.0,
        "avg_arpu": (df.groupby("CustomerID", observed=False)["ARPU"].mean().mean()
                     if "ARPU" in df.columns and not df.empty else 0.0),
        "total_lift": df["Price_Difference"].sum() if "Price_Difference" in df.columns else 0.0,
        "avg_lift": df["Price_Difference"].mean() if "Price_Difference" in df.columns else 0.0,
        "positive_lift_pct": ((df["Price_Difference"] > 0).mean() * 100) if "Price_Difference" in df.columns else 0.0,
        "month_arpu": month_df["ARPU"].sum() if selected_month and "ARPU" in df.columns else 0.0,
        "month_projected": ((month_df["ARPU"] + month_df["Price_Difference"]).sum()
                            if selected_month and {"ARPU","Price_Difference"}.issubset(df.columns) else 0.0),
    }
    return metrics

def create_arpu_chart(df: pd.DataFrame, selected_month: int | None = None):
    hist_arpu = df.groupby("MONTH", observed=False)["ARPU"].sum().reset_index()
    hist_arpu["Month_Name"] = hist_arpu["MONTH"].map(MONTH_ABBR)
    proj_arpu = df.groupby("MONTH", observed=False).agg({"ARPU": "sum", "Price_Difference": "sum"}).reset_index()
    proj_arpu["Projected_ARPU"] = proj_arpu["ARPU"] + proj_arpu["Price_Difference"]
    proj_arpu["Month_Name"] = proj_arpu["MONTH"].map(MONTH_ABBR)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist_arpu["Month_Name"], y=hist_arpu["ARPU"],
        mode="lines+markers", name="Actual ARPU 2024",
        line=dict(color="#1A73E8", width=3), marker=dict(size=8)
    ))
    fig.add_trace(go.Scatter(
        x=proj_arpu["Month_Name"], y=proj_arpu["Projected_ARPU"],
        mode="lines+markers", name="Projected ARPU 2025 (M-Series Transition)",
        line=dict(color="#25D366", width=3, dash="dash"), marker=dict(size=8)
    ))

    if selected_month:
        month_name = MONTH_ABBR[selected_month]
        month_idx = list(MONTH_ABBR.values()).index(month_name)
        fig.add_vrect(x0=month_idx - 0.4, x1=month_idx + 0.4,
                      fillcolor="#E0E0E0", opacity=0.3, layer="below", line_width=0)

    fig.update_layout(
        title="ARPU Trends: Historical vs Projected Impact",
        xaxis_title="Month", yaxis_title="ARPU ($)",
        template="plotly_white", height=400, hovermode="x unified",
        showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def create_persona_distribution(df: pd.DataFrame, persona_type="Liquidity_Persona"):
    base = df.drop_duplicates("CustomerID")
    if persona_type not in base.columns or base.empty:
        return go.Figure()
    dist = base[persona_type].value_counts()
    fig = go.Figure(data=[go.Bar(
        x=dist.index, y=dist.values, marker_color="#1A73E8",
        text=dist.values, textposition="auto"
    )])
    fig.update_layout(
        title=f"Customer Distribution by {persona_type.replace('_', ' ')}",
        xaxis_title=persona_type.replace("_", " "), yaxis_title="Number of Customers",
        template="plotly_white", height=350, showlegend=False
    )
    return fig

def create_lift_distribution(df: pd.DataFrame, liquidity=None, consumption=None):
    filtered_df = df
    if liquidity and liquidity != "All" and "Liquidity_Persona" in df.columns:
        filtered_df = filtered_df[filtered_df["Liquidity_Persona"] == liquidity]
    if consumption and consumption != "All" and "Consumption_Persona" in df.columns:
        filtered_df = filtered_df[filtered_df["Consumption_Persona"] == consumption]
    if "Price_Difference" not in filtered_df.columns:
        return go.Figure()
    lift_by_month = filtered_df.groupby("MONTH", observed=False)["Price_Difference"].mean().reset_index()
    lift_by_month["Month_Name"] = lift_by_month["MONTH"].map(MONTH_ABBR)
    colors = ["#25D366" if x > 0 else "#FF6B6B" for x in lift_by_month["Price_Difference"]]

    fig = go.Figure(data=[go.Bar(
        x=lift_by_month["Month_Name"], y=lift_by_month["Price_Difference"],
        marker_color=colors, text=[f"${x:.2f}" for x in lift_by_month["Price_Difference"]], textposition="auto"
    )])
    fig.update_layout(
        title="Average ARPU Lift by Month", xaxis_title="Month", yaxis_title="ARPU Lift ($)",
        template="plotly_white", height=350, showlegend=False
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    return fig

def create_sub_persona_chart(df: pd.DataFrame):
    base = df.drop_duplicates("CustomerID")
    if "Sub_Persona" not in base.columns or base.empty:
        return go.Figure()
    sub_dist = base["Sub_Persona"].value_counts().head(10)
    fig = go.Figure(data=[go.Pie(
        labels=sub_dist.index, values=sub_dist.values, hole=0.4,
        marker=dict(colors=px.colors.qualitative.Set3[:len(sub_dist)])
    )])
    fig.update_layout(
        title="Customer Distribution by Sub-Persona", template="plotly_white",
        height=350, showlegend=True, legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05)
    )
    return fig

def create_device_distribution(df: pd.DataFrame):
    base = df.drop_duplicates("CustomerID")
    if "Device_Category" not in base.columns or base.empty:
        return go.Figure()
    device_dist = base["Device_Category"].value_counts()
    total = device_dist.sum() or 1
    percentages = (device_dist / total * 100).round(1)
    fig = go.Figure(data=[go.Bar(
        x=device_dist.index, y=device_dist.values,
        marker_color=["#1A73E8", "#4285F4", "#669DF6", "#AECBFA"][:len(device_dist)],
        text=[f"{device_dist[cat]}<br>({percentages[cat]}%)" for cat in device_dist.index],
        textposition="auto"
    )])
    fig.update_layout(
        title="Device Category Distribution", xaxis_title="Device Category", yaxis_title="Number of Customers",
        template="plotly_white", height=350, showlegend=False
    )
    return fig

def create_persona_matrix(df: pd.DataFrame):
    if not {"Liquidity_Persona", "Consumption_Persona"}.issubset(df.columns):
        return go.Figure()
    persona_counts = (
        df.drop_duplicates("CustomerID")
          .groupby(["Liquidity_Persona", "Consumption_Persona"], observed=False)
          .size()
          .reset_index(name="Count")
    )
    if persona_counts.empty:
        return go.Figure()
    persona_pivot = persona_counts.pivot(index="Liquidity_Persona", columns="Consumption_Persona", values="Count").fillna(0)
    fig = go.Figure(data=go.Heatmap(
        z=persona_pivot.values, x=persona_pivot.columns, y=persona_pivot.index,
        colorscale="Blues", text=persona_pivot.values.astype(int), texttemplate="%{text}",
        textfont={"size": 12}, colorbar=dict(title="Customer Count")
    ))
    fig.update_layout(
        title="Customer Count by Persona Combination", xaxis_title="Consumption Persona",
        yaxis_title="Liquidity Persona", height=450, template="plotly_white"
    )
    return fig

def create_persona_lift_heatmap(df: pd.DataFrame):
    """Heatmap with correct values per cell and 2-decimal labels."""
    if not {"Liquidity_Persona", "Consumption_Persona", "Price_Difference"}.issubset(df.columns):
        return go.Figure()

    persona_lift = (
        df.groupby(["Liquidity_Persona", "Consumption_Persona"], observed=False)["Price_Difference"]
          .mean()
          .reset_index()
    )
    persona_pivot = persona_lift.pivot(index="Liquidity_Persona", columns="Consumption_Persona", values="Price_Difference")

    # Build a string matrix for labels (ensures 2-dec and avoids plotly reformatting)
    text_fmt = persona_pivot.applymap(lambda v: "" if pd.isna(v) else f"{v:.2f}")

    # Percentile-based color scaling to handle outliers
    values_flat = persona_pivot.values.flatten()
    values_flat = values_flat[~np.isnan(values_flat)]
    if len(values_flat) > 0:
        vmin = float(np.percentile(values_flat, 5))
        vmax = float(np.percentile(values_flat, 95))
    else:
        vmin, vmax = 0.0, 10.0

    fig = go.Figure(data=go.Heatmap(
        z=persona_pivot.values,
        x=persona_pivot.columns,
        y=persona_pivot.index,
        colorscale=[
            [0, "#FF4444"], [0.25, "#FFB6B6"], [0.5, "#FFFFFF"],
            [0.75, "#B6FFB6"], [1, "#44FF44"]
        ],
        zmid=0,
        zmin=vmin,
        zmax=vmax,
        text=text_fmt.values,
        texttemplate="$%{text}",   # uses the pre-formatted strings
        textfont={"size": 10},
        colorbar=dict(title="Avg ARPU Lift ($)")
    ))
    fig.update_layout(
        title="Average ARPU Lift by Persona Combination",
        xaxis_title="Consumption Persona",
        yaxis_title="Liquidity Persona",
        height=450,
        template="plotly_white"
    )
    return fig

def calculate_persona_metrics(df: pd.DataFrame, liquidity=None, consumption=None):
    filtered_df = df
    if liquidity and liquidity != "All" and "Liquidity_Persona" in df.columns:
        filtered_df = filtered_df[filtered_df["Liquidity_Persona"] == liquidity]
    if consumption and consumption != "All" and "Consumption_Persona" in df.columns:
        filtered_df = filtered_df[filtered_df["Consumption_Persona"] == consumption]

    if filtered_df.empty:
        return {
            "total_customers": 0, "total_annual_mbs": 0.0, "avg_annual_mbs": 0.0,
            "total_voice": 0.0, "avg_voice": 0.0, "total_annual_spend": 0.0,
            "avg_monthly_spend": 0.0, "spend_volatility": 0.0,
            "arpu_lift_sum": 0.0, "arpu_pct_of_total": 0.0, "arpu_lift_pct_of_total": 0.0,
            "total_data_to_voice": 0.0, "avg_data_to_voice": 0.0
        }, {}, {}

    unique_customers = filtered_df.drop_duplicates("CustomerID")

    def _safe_group_mean(col, agg="mean"):
        if col in filtered_df.columns:
            g = filtered_df.groupby("CustomerID", observed=False)[col]
            return getattr(g, "sum" if agg == "sum" else "mean")().mean()
        return 0.0

    total_annual_mbs = float(filtered_df["MB_CONSUMPTION"].sum()) if "MB_CONSUMPTION" in filtered_df.columns else 0.0
    avg_annual_mbs = _safe_group_mean("MB_CONSUMPTION", agg="sum")
    total_voice = float(filtered_df["MINUTES"].sum()) if "MINUTES" in filtered_df.columns else 0.0
    avg_voice = _safe_group_mean("MINUTES", agg="sum")
    total_annual_spend = float(filtered_df["ARPU"].sum()) if "ARPU" in filtered_df.columns else 0.0
    avg_monthly_spend = _safe_group_mean("ARPU", agg="mean")
    spend_volatility = (filtered_df.groupby("CustomerID", observed=False)["ARPU"].std().mean()
                        if "ARPU" in filtered_df.columns and not filtered_df.empty else 0.0)
    arpu_lift_sum = float(filtered_df["Price_Difference"].sum()) if "Price_Difference" in filtered_df.columns else 0.0

    # Data/voice ratios
    if total_voice > 0:
        total_data_to_voice = total_annual_mbs / total_voice
        avg_data_to_voice = filtered_df.apply(
            lambda row: (row["MB_CONSUMPTION"] / row["MINUTES"]) if row.get("MINUTES", 0) else 0, axis=1
        ).mean()
    else:
        total_data_to_voice = 0.0
        avg_data_to_voice = 0.0

    device_comp = unique_customers["Device_Category"].value_counts().to_dict() if "Device_Category" in unique_customers.columns else {}
    sub_persona_comp = unique_customers["Sub_Persona"].value_counts().head(10).to_dict() if "Sub_Persona" in unique_customers.columns else {}

    metrics = {
        "total_customers": int(unique_customers.shape[0]),
        "total_annual_mbs": total_annual_mbs,
        "avg_annual_mbs": float(avg_annual_mbs),
        "total_voice": total_voice,
        "avg_voice": float(avg_voice),
        "total_annual_spend": total_annual_spend,
        "avg_monthly_spend": float(avg_monthly_spend),
        "spend_volatility": float(spend_volatility),
        "arpu_lift_sum": arpu_lift_sum,
        "arpu_pct_of_total": 0.0,
        "arpu_lift_pct_of_total": 0.0,
        "total_data_to_voice": float(total_data_to_voice),
        "avg_data_to_voice": float(avg_data_to_voice),
    }
    return metrics, device_comp, sub_persona_comp

# =============================================================================
# Main app
# =============================================================================
def main():
    # Auth
    if not check_password():
        return

    # Header
    col1, col2 = st.columns([1, 5])
    with col1:
        try:
            logo = Image.open("touch.png")
            st.image(logo, width=120)
        except Exception:
            st.markdown("### 📱 TOUCH")
    with col2:
        st.title("RECOMMENDATION ENGINE DASHBOARD")
        st.markdown("*Personalized Offer Optimization Platform using 2-Way Clustering*")

    # Load data
    with st.spinner("Loading data..."):
        df = load_data()
    if df is None or df.empty:
        st.error("Failed to load data from Google Drive. Ensure sharing is 'Anyone with the link can view', then click Rerun.")
        st.stop()

    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    customer_type_options = ["All"] + (sorted(df["Customer_Type"].dropna().unique().tolist())
                                       if "Customer_Type" in df.columns else [])
    selected_customer_type = st.sidebar.selectbox("Customer Type", customer_type_options, index=0)

    liquidity_options = ["All"] + (sorted(df["Liquidity_Persona"].dropna().unique().tolist())
                                   if "Liquidity_Persona" in df.columns else [])
    selected_liquidity = st.sidebar.selectbox("Liquidity Persona", liquidity_options, index=0)

    consumption_options = ["All"] + (sorted(df["Consumption_Persona"].dropna().unique().tolist())
                                     if "Consumption_Persona" in df.columns else [])
    selected_consumption = st.sidebar.selectbox("Consumption Persona", consumption_options, index=0)

    selected_month = st.sidebar.slider("Select Month", min_value=1, max_value=12, value=12, format="%d")
    st.sidebar.markdown(f"**Selected:** {MONTH_NAMES[selected_month]}")

    # Apply filters
    filtered_df = df
    if selected_customer_type != "All" and "Customer_Type" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Customer_Type"] == selected_customer_type]
    if selected_liquidity != "All" and "Liquidity_Persona" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Liquidity_Persona"] == selected_liquidity]
    if selected_consumption != "All" and "Consumption_Persona" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Consumption_Persona"] == selected_consumption]

    # Metrics
    metrics = create_kpi_metrics(filtered_df, selected_month)
    total_company_arpu = df["ARPU"].sum() if "ARPU" in df.columns else 0.0
    total_company_projected = (df["ARPU"] + df["Price_Difference"]).sum() if {"ARPU","Price_Difference"}.issubset(df.columns) else 0.0

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Recommendation Overview", "🔍 Customer Explorer", "📈 Analytics", "🎯 Persona Deep Dive", "📋 Offer Catalog"
    ])

    # --- Tab 1: Overview
    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Customers", f"{metrics['total_customers']:,}")
        c2.metric("Total Annual ARPU", f"${metrics['total_arpu']:,.0f}")
        c3.metric("Projected Total ARPU Lift", f"${metrics['total_lift']:,.0f}",
                  delta=f"{metrics['positive_lift_pct']:.1f}% positive", delta_color="normal")
        c4.metric("Avg ARPU Lift per Customer", f"${metrics['avg_lift']:.2f}")

        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"ARPU - {MONTH_ABBR[selected_month]}", f"${metrics['month_arpu']:,.0f}")
        c2.metric(f"Projected - {MONTH_ABBR[selected_month]}", f"${metrics['month_projected']:,.0f}",
                  delta=f"+${metrics['month_projected'] - metrics['month_arpu']:,.0f}", delta_color="normal")
        month_df = filtered_df[filtered_df["MONTH"] == selected_month] if "MONTH" in filtered_df.columns else pd.DataFrame()
        active_customers = month_df["CustomerID"].nunique() if "CustomerID" in month_df.columns else 0
        c3.metric(f"Active Customers - {MONTH_ABBR[selected_month]}", f"{active_customers:,}")
        avg_month_lift = month_df["Price_Difference"].mean() if "Price_Difference" in month_df.columns and not month_df.empty else 0
        c4.metric(f"Avg Lift - {MONTH_ABBR[selected_month]}", f"${avg_month_lift:.2f}")

        st.markdown("---")
        a, b = st.columns(2)
        with a:
            st.plotly_chart(create_arpu_chart(filtered_df, selected_month), use_container_width=True)
        with b:
            st.plotly_chart(create_lift_distribution(filtered_df, selected_liquidity, selected_consumption), use_container_width=True)

        a, b = st.columns(2)
        with a:
            st.plotly_chart(create_persona_matrix(filtered_df), use_container_width=True)
        with b:
            st.plotly_chart(create_persona_lift_heatmap(filtered_df), use_container_width=True)

        a, b = st.columns(2)
        with a:
            persona_type = st.selectbox("Select Persona View", ["Liquidity_Persona", "Consumption_Persona"], label_visibility="collapsed")
            st.plotly_chart(create_persona_distribution(filtered_df, persona_type), use_container_width=True)
        with b:
            st.plotly_chart(create_device_distribution(filtered_df), use_container_width=True)

        a, b = st.columns(2)
        with a:
            st.plotly_chart(create_sub_persona_chart(filtered_df), use_container_width=True)
        with b:
            st.markdown("### 📊 Quick Insights")
            try:
                top_sub_persona = filtered_df.drop_duplicates("CustomerID")["Sub_Persona"].value_counts().index[0]
            except Exception:
                top_sub_persona = "N/A"
            avg_consumption = filtered_df["MB_CONSUMPTION"].mean() if "MB_CONSUMPTION" in filtered_df.columns else 0
            try:
                top_device = filtered_df.drop_duplicates("CustomerID")["Device_Category"].value_counts().index[0]
            except Exception:
                top_device = "N/A"
            st.info(f"""
**Key Findings:**
- Most common sub-persona: **{top_sub_persona}**
- Average data consumption: **{avg_consumption:,.0f} MB**
- Dominant device category: **{top_device}**
- Customers with positive lift: **{metrics['positive_lift_pct']:.1f}%**
- Total projected ARPU increase: **${metrics['total_lift']:,.0f}**
""")

    # --- Tab 2: Customer Explorer (hardened)
    with tab2:
        st.markdown("### 🔍 Customer Explorer")
        col_left, col_right = st.columns([1, 3])

        with col_left:
            st.markdown("#### Search & Filter")

            search_term = st.text_input("Search Customer ID", placeholder="Enter Customer ID...")

            explorer_customer_type = st.selectbox(
                "Customer Type",
                ["All"] + (sorted(filtered_df["Customer_Type"].dropna().unique().tolist())
                           if "Customer_Type" in filtered_df.columns else []),
                key="explorer_customer_type",
            )
            explorer_device = st.selectbox(
                "Device Category",
                ["All"] + (sorted(filtered_df["Device_Category"].dropna().unique().tolist())
                           if "Device_Category" in filtered_df.columns else [])
            )

            explorer_df = filtered_df
            if explorer_customer_type != "All" and "Customer_Type" in explorer_df.columns:
                explorer_df = explorer_df[explorer_df["Customer_Type"] == explorer_customer_type]
            if explorer_device != "All" and "Device_Category" in explorer_df.columns:
                explorer_df = explorer_df[explorer_df["Device_Category"] == explorer_device]

            if "CustomerID" in explorer_df.columns:
                all_customer_ids = sorted(map(str, explorer_df["CustomerID"].dropna().unique()))
            else:
                all_customer_ids = []

            if search_term:
                q = search_term.lower()
                filtered_customer_ids = [cid for cid in all_customer_ids if q in cid.lower()]
            else:
                filtered_customer_ids = all_customer_ids

            st.markdown("#### Customer List")
            items_per_page = 50
            total_filtered = len(filtered_customer_ids)
            total_pages = max(1, (total_filtered + items_per_page - 1) // items_per_page)

            if "explorer_page" not in st.session_state:
                st.session_state.explorer_page = 1
            st.session_state.explorer_page = min(max(1, st.session_state.explorer_page), total_pages)

            page_col1, page_col2, page_col3 = st.columns([1, 2, 1])
            with page_col2:
                current_page = st.number_input(
                    "Page",
                    min_value=1,
                    max_value=total_pages,
                    value=st.session_state.explorer_page,
                    step=1,
                    label_visibility="collapsed",
                    key="explorer_page_input",
                )
            if current_page != st.session_state.explorer_page:
                st.session_state.explorer_page = int(current_page)

            start_idx = (st.session_state.explorer_page - 1) * items_per_page
            end_idx = min(start_idx + items_per_page, total_filtered)
            page_customer_ids = filtered_customer_ids[start_idx:end_idx]

            selected_customer = None
            if page_customer_ids:
                selected_customer = st.selectbox(
                    "Select Customer",
                    page_customer_ids,
                    key="customer_select",       # stable key (prevents state clashes)
                    label_visibility="collapsed",
                )
            else:
                st.warning("No customers found matching the search criteria")

            if search_term:
                st.info(f"Found {total_filtered:,} customers matching '{search_term}'")
                if total_pages > 1:
                    st.info(f"Page {st.session_state.explorer_page} of {total_pages} | "
                            f"Showing {start_idx+1}-{end_idx} of {total_filtered:,}")
            else:
                st.info(f"Total: {len(all_customer_ids):,} customers")
                if total_pages > 1:
                    st.info(f"Page {st.session_state.explorer_page} of {total_pages} | "
                            f"Showing {start_idx+1}-{end_idx}")

        with col_right:
            try:
                if selected_customer:
                    if {"CustomerID"}.issubset(explorer_df.columns):
                        cust_mask = (explorer_df["CustomerID"].astype(str) == str(selected_customer))
                        if "MONTH" in explorer_df.columns:
                            cust_mask &= (explorer_df["MONTH"] == selected_month)
                        cust_data = explorer_df.loc[cust_mask]
                    else:
                        cust_data = pd.DataFrame()

                    if cust_data.empty:
                        st.warning(f"No data available for customer {selected_customer} in {MONTH_NAMES[selected_month]}")
                    else:
                        row = cust_data.iloc[0]

                        st.markdown(f"### Customer: {selected_customer}")
                        cA, cB, cC, cD = st.columns(4)
                        cA.markdown(f"**Type:** {row.get('Customer_Type', 'N/A')}")
                        cB.markdown(f"**Liquidity:** {row.get('Liquidity_Persona', 'N/A')}")
                        cC.markdown(f"**Consumption:** {row.get('Consumption_Persona', 'N/A')}")
                        cD.markdown(f"**Sub-Persona:** {row.get('Sub_Persona', 'N/A')}")

                        st.markdown("---")
                        st.markdown(f"#### Current Status - {MONTH_NAMES[selected_month]}")
                        c1, c2 = st.columns(2)
                        with c1:
                            current_offers = row.get("offer_pattern_str", "None")
                            st.markdown(f"**Current Offers:** {current_offers}")
                            mb_consumption = row.get("MB_CONSUMPTION", 0) or 0
                            mb_allowance   = row.get("mb_allowance", 0) or 0
                            mb_usage_pct   = row.get("mb_usage_pct", 0) or 0
                            st.markdown(f"**Data Usage:** {mb_consumption:,.0f} MB / {mb_allowance:,.0f} MB ({mb_usage_pct:.1f}%)")
                        with c2:
                            arpu = row.get("ARPU", 0) or 0
                            st.markdown(f"**Current ARPU:** ${arpu:.2f}")
                            minutes = row.get("MINUTES", 0) or 0
                            st.markdown(f"**Voice Minutes:** {minutes:,.0f}")

                        st.markdown("---")
                        st.markdown("#### Recommendation")
                        recommended = row.get("recommended_offers_str", "None")
                        if recommended and recommended != "None":
                            st.success(f"**Recommended Plan:** {recommended}")
                        else:
                            st.warning("**No recommendation available**")

                        c1, c2 = st.columns(2)
                        price_diff = row.get("Price_Difference", 0) or 0
                        color = "green" if price_diff > 0 else "red" if price_diff < 0 else "gray"
                        c1.markdown(
                            f"**ARPU Lift:** <span style='color:{color}; font-weight:bold;'>${price_diff:.2f}</span>",
                            unsafe_allow_html=True
                        )
                        new_price = row.get("Recommended_Offer_Price", 0) or 0
                        c2.markdown(f"**New Price:** ${new_price:.2f}")

                        message = row.get("Message_English", "")
                        if isinstance(message, str) and message.strip():
                            st.info(f"📱 {message}")

                        st.markdown("---")
                        st.markdown("#### Customer History (All Months)")
                        if {"CustomerID","MONTH","ARPU","Price_Difference"}.issubset(explorer_df.columns):
                            cust_history = explorer_df[explorer_df["CustomerID"].astype(str) == str(selected_customer)] \
                                            .sort_values("MONTH")
                            fig = go.Figure()
                            if not cust_history.empty:
                                fig.add_trace(go.Scatter(
                                    x=cust_history["MONTH"].map(MONTH_ABBR),
                                    y=cust_history["ARPU"],
                                    mode="lines+markers", name="Actual ARPU",
                                    line=dict(color="#1A73E8", width=2)
                                ))
                                fig.add_trace(go.Scatter(
                                    x=cust_history["MONTH"].map(MONTH_ABBR),
                                    y=cust_history["ARPU"] + cust_history["Price_Difference"],
                                    mode="lines+markers", name="Potential ARPU",
                                    line=dict(color="#25D366", width=2, dash="dash")
                                ))
                            fig.update_layout(title="Customer ARPU Trend", height=300, template="plotly_white", showlegend=True)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Not enough columns to draw history.")
            except Exception as e:
                st.error("Customer Explorer hit an error. Details below:")
                st.exception(e)

    # --- Tab 3: Analytics (Offer Migration + Stats fixes)
    with tab3:
        st.markdown("### 📈 Advanced Analytics")
        st.markdown("#### Offer Migration Analysis")

        # Current offers: drop 'None' and 'NO_RECOMMENDATION' explicitly
        if "offer_pattern_str" in filtered_df.columns:
            cur_mask = ~filtered_df["offer_pattern_str"].apply(_is_no_rec_string)
            current_offers_filtered = filtered_df.loc[cur_mask, "offer_pattern_str"].value_counts().head(10)
        else:
            current_offers_filtered = pd.Series(dtype=int)

        # Recommended offers: use recommended_has, which already excludes 'None'/'NO_RECOMMENDATION'
        if "recommended_has" in filtered_df.columns and "recommended_offers_str" in filtered_df.columns:
            rec_mask = filtered_df["recommended_has"]
            recommended_offers_filtered = filtered_df.loc[rec_mask, "recommended_offers_str"].value_counts().head(10)
        else:
            recommended_offers_filtered = pd.Series(dtype=int)

        a, b = st.columns(2)
        with a:
            st.markdown("**Top Current Offers**")
            if not current_offers_filtered.empty:
                fig = go.Figure(data=[go.Bar(
                    y=current_offers_filtered.index,
                    x=current_offers_filtered.values,
                    orientation="h",
                    marker_color="#4285F4",
                    text=current_offers_filtered.values,
                    textposition="auto",
                )])
                fig.update_layout(
                    height=400,
                    template="plotly_white",
                    showlegend=False,
                    margin=dict(l=10, r=10, t=40, b=10),
                )
                # Hide x-axis completely as requested
                fig.update_xaxes(visible=False, showgrid=False, zeroline=False)
                fig.update_yaxes(showgrid=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No current offers to display")

        with b:
            st.markdown("**Top Recommended Offers**")
            if not recommended_offers_filtered.empty:
                fig = go.Figure(data=[go.Bar(
                    y=recommended_offers_filtered.index,
                    x=recommended_offers_filtered.values,
                    orientation="h",
                    marker_color="#25D366",
                    text=recommended_offers_filtered.values,
                    textposition="auto",
                )])
                fig.update_layout(
                    height=400,
                    template="plotly_white",
                    showlegend=False,
                    margin=dict(l=10, r=10, t=40, b=10),
                )
                fig.update_xaxes(visible=False, showgrid=False, zeroline=False)
                fig.update_yaxes(showgrid=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No recommended offers to display")

        st.markdown("---")
        st.markdown("#### Summary Statistics")
        a, b, c = st.columns(3)
        with a:
            st.markdown("**Data Coverage**")
            total_months = filtered_df["MONTH"].nunique() if "MONTH" in filtered_df.columns else 0
            total_records = len(filtered_df)
            avg_records_per_customer = (filtered_df.groupby("CustomerID", observed=False).size().mean()
                                        if "CustomerID" in filtered_df.columns and total_records > 0 else 0)
            st.metric("Total Months", total_months)
            st.metric("Total Records", f"{total_records:,}")
            st.metric("Avg Records/Customer", f"{avg_records_per_customer:.1f}")

        with b:
            st.markdown("**Recommendation Impact**")
            # Robust 'no recommendation' count
            if "recommended_offers_str" in filtered_df.columns:
                no_rec_count = int(filtered_df["recommended_offers_str"].apply(_is_no_rec_string).sum())
                rec_rate = (100.0 * (1 - no_rec_count / len(filtered_df))) if len(filtered_df) > 0 else 0.0
            else:
                no_rec_count = len(filtered_df)
                rec_rate = 0.0
            positive_lift = (filtered_df["Price_Difference"] > 0).sum() if "Price_Difference" in filtered_df.columns else 0

            st.metric("Recommendation Rate", f"{rec_rate:.1f}%")
            st.metric("Positive Lift Cases", f"{positive_lift:,}")
            st.metric("No Recommendation", f"{no_rec_count:,}")

        with c:
            st.markdown("**Financial Impact**")
            total_current = filtered_df["ARPU"].sum() if "ARPU" in filtered_df.columns else 0.0
            total_recommended = ((filtered_df["ARPU"] + filtered_df["Price_Difference"]).sum()
                                 if {"ARPU", "Price_Difference"}.issubset(filtered_df.columns) else total_current)
            growth_rate = ((total_recommended - total_current) / total_current * 100) if total_current > 0 else 0
            st.metric("Current Total ARPU", f"${total_current:,.0f}")
            st.metric("Projected Total ARPU", f"${total_recommended:,.0f}")
            st.metric("Growth Rate", f"{growth_rate:.2f}%")

    # --- Tab 4: Persona Deep Dive
    with tab4:
        st.markdown("### 🎯 Persona Deep Dive")
        st.markdown("*Change the filters to explore different persona combinations*")

        persona_metrics, device_comp, sub_persona_comp = calculate_persona_metrics(
            filtered_df, selected_liquidity, selected_consumption
        )
        persona_metrics["arpu_pct_of_total"] = ((persona_metrics["total_annual_spend"] / total_company_arpu * 100)
                                                if total_company_arpu > 0 else 0.0)
        persona_metrics["arpu_lift_pct_of_total"] = (((persona_metrics["total_annual_spend"] + persona_metrics["arpu_lift_sum"])
                                                      / total_company_projected * 100)
                                                     if total_company_projected > 0 else 0.0)

        st.markdown("#### 📊 Key Metrics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Customers", f"{persona_metrics['total_customers']:,}")
        c1.metric("Total Annual MBs", f"{persona_metrics['total_annual_mbs']:,.0f}")
        c1.metric("Avg Annual MBs", f"{persona_metrics['avg_annual_mbs']:,.0f}")

        c2.metric("Total Voice Minutes", f"{persona_metrics['total_voice']:,.0f}")
        c2.metric("Avg Voice Minutes", f"{persona_metrics['avg_voice']:,.0f}")
        c2.metric("Data/Voice Ratio", f"{persona_metrics.get('avg_data_to_voice', 0):.2f}")

        c3.metric("Total Annual Spend", f"${persona_metrics['total_annual_spend']:,.0f}")
        c3.metric("Avg Monthly Spend", f"${persona_metrics['avg_monthly_spend']:.2f}")
        c3.metric("Spend Volatility", f"${persona_metrics['spend_volatility']:.2f}")

        c4.metric("ARPU Lift Sum", f"${persona_metrics['arpu_lift_sum']:,.0f}")
        c4.metric("% of Total ARPU", f"{persona_metrics['arpu_pct_of_total']:.1f}%")
        c4.metric("% of Projected ARPU", f"{persona_metrics['arpu_lift_pct_of_total']:.1f}%")

        st.markdown("---")
        a, b = st.columns(2)
        with a:
            st.markdown("#### 📱 Device Composition")
            if device_comp:
                device_df = pd.DataFrame(list(device_comp.items()), columns=["Device", "Count"])
                device_df["Percentage"] = (device_df["Count"] / device_df["Count"].sum() * 100).round(1)
                fig = go.Figure(data=[go.Pie(labels=device_df["Device"], values=device_df["Count"],
                                             hole=0.4, textinfo="label+percent",
                                             marker=dict(colors=px.colors.qualitative.Set3[:len(device_df)]))])
                fig.update_layout(height=350, showlegend=True, template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No device data available")
        with b:
            st.markdown("#### 👥 Sub-Persona Composition")
            if sub_persona_comp:
                sub_df = pd.DataFrame(list(sub_persona_comp.items()), columns=["Sub-Persona", "Count"]).sort_values("Count", ascending=True)
                fig = go.Figure(data=[go.Bar(y=sub_df["Sub-Persona"], x=sub_df["Count"], orientation="h",
                                             marker_color="#1A73E8", text=sub_df["Count"], textposition="auto")])
                fig.update_layout(height=350, xaxis_title="Number of Customers", template="plotly_white", showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No sub-persona data available")

        st.markdown("---")
        st.markdown("#### 💡 Persona Insights")
        filters_text = []
        if selected_customer_type != "All":
            filters_text.append(f"Customer Type: **{selected_customer_type}**")
        if selected_liquidity != "All":
            filters_text.append(f"Liquidity: **{selected_liquidity}**")
        if selected_consumption != "All":
            filters_text.append(f"Consumption: **{selected_consumption}**")
        st.info("Current filters: " + " | ".join(filters_text) if filters_text else "Showing data for **All Personas**")

        a, b = st.columns(2)
        a.markdown("##### Persona Insights")
        a.write(f"- **Avg Data Usage:** {persona_metrics['avg_annual_mbs']/12:.0f} MB/month")
        a.write(f"- **Avg Voice Usage:** {persona_metrics['avg_voice']/12:.0f} minutes/month")
        lift_per_cust = (persona_metrics["arpu_lift_sum"]/persona_metrics["total_customers"]) if persona_metrics["total_customers"] > 0 else 0.0
        b.markdown("##### Business Impact")
        b.write(f"- **Revenue Contribution:** {persona_metrics['arpu_pct_of_total']:.1f}% of total ARPU")
        b.write(f"- **Growth Potential:** ${persona_metrics['arpu_lift_sum']:,.0f} total lift opportunity")
        b.write(f"- **Lift per Customer:** ${lift_per_cust:.2f} average")
        b.write(f"- **Customer Base:** {persona_metrics['total_customers']:,} customers")
        b.write(f"- **Post-Optimization Share:** {persona_metrics['arpu_lift_pct_of_total']:.1f}% of projected ARPU")

    # --- Tab 5: Offer Catalog
    with tab5:
        st.markdown("### 📋 Offer Catalog")
        st.markdown("Browse all available offers organized by category")
        offer_tabs = st.tabs(list(OFFER_CATALOG.keys()))
        for i, (category, df_offers) in enumerate(OFFER_CATALOG.items()):
            with offer_tabs[i]:
                st.markdown(f"#### {category}")
                display_df = df_offers.copy()
                display_df["Data (GB)"] = (display_df["Data (MB)"] / 1024).round(2)
                display_df = display_df[["Code", "Name", "Type", "Data (MB)", "Data (GB)", "Price ($)", "Voice"]]
                st.dataframe(
                    display_df, use_container_width=True, hide_index=True,
                    column_config={
                        "Code": st.column_config.TextColumn("Offer Code", width="medium"),
                        "Name": st.column_config.TextColumn("Offer Name", width="medium"),
                        "Type": st.column_config.TextColumn("Type", width="small"),
                        "Data (MB)": st.column_config.NumberColumn("Data (MB)", format="%d"),
                        "Data (GB)": st.column_config.NumberColumn("Data (GB)", format="%.2f"),
                        "Price ($)": st.column_config.NumberColumn("Price ($)", format="$%.2f"),
                        "Voice": st.column_config.NumberColumn("Voice Minutes", format="%d"),
                    }
                )
                a, b, c = st.columns(3)
                a.metric("Total Offers", len(display_df))
                b.metric("Avg Price", f"${display_df['Price ($)'].mean():.2f}")
                c.metric("Avg Data", f"{display_df['Data (MB)'].mean():,.0f} MB")

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #5F6368; padding: 20px;'>"
        "<small>Touch Recommendation Dashboard v1.0 | Data as of 2024 | Built with Streamlit</small>"
        "</div>",
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
