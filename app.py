import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import ast
from PIL import Image
import os
import hashlib
import urllib.request
import ssl

# Page config
st.set_page_config(
    page_title="Touch Recommendation Dashboard",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
MONTH_NAMES = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December"
}
MONTH_ABBR = {m: MONTH_NAMES[m][:3] for m in MONTH_NAMES}

# Password configuration
PASSWORD = "msba@touch"

# Google Drive file configuration
GDRIVE_FILE_ID = "1cPFjQmRzuZhwq0Q1S4l8fihhWwHhIcQs"
DATA_FILE_NAME = "data.parquet"

# Offer catalog data
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

# Function to download from Google Drive
@st.cache_data(persist=True, show_spinner=False)
def download_from_gdrive():
    """Download parquet file from Google Drive - completely free method"""
    try:
        # Check if file already exists
        if os.path.exists(DATA_FILE_NAME):
            return pd.read_parquet(DATA_FILE_NAME)
        
        # Create download URL
        download_url = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"
        
        # Download the file with progress
        with st.spinner("Downloading data file (one-time download, ~35MB)..."):
            # Create SSL context to handle certificates
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            # Download file
            with urllib.request.urlopen(download_url, context=ssl_context) as response:
                with open(DATA_FILE_NAME, 'wb') as out_file:
                    out_file.write(response.read())
        
        # Load the parquet file
        df = pd.read_parquet(DATA_FILE_NAME)
        st.success("✅ Data loaded successfully!")
        return df
        
    except Exception as e:
        st.error(f"Error downloading data: {str(e)}")
        st.info("Please make sure the Google Drive link is set to 'Anyone with the link can view'")
        return None

# Modified load_data function
@st.cache_data(persist=True, show_spinner=False)
def load_data():
    """Load and preprocess the data with caching"""
    try:
        # Download from Google Drive
        df = download_from_gdrive()
        
        if df is None:
            return None
        
        # Memory optimization - convert object columns to category where appropriate
        categorical_columns = ['Customer_Type', 'Liquidity_Persona', 'Consumption_Persona', 
                              'Sub_Persona', 'Device_Category', 'DEVICE_MODEL']
        
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].astype('category')
        
        # Convert CustomerID to string to save memory
        if 'CustomerID' in df.columns:
            df['CustomerID'] = df['CustomerID'].astype(str)
        
        # Preprocess data
        df['MONTH'] = pd.to_numeric(df['MONTH'], errors='coerce').fillna(0).astype(int)
        df = df[df['MONTH'].between(1, 12)]
        
        # Parse offer patterns
        df['offer_pattern_norm'] = df['offer_pattern'].apply(normalize_offer_pattern)
        df['offer_pattern_str'] = df['offer_pattern_norm'].apply(
            lambda lst: ", ".join(lst) if lst else "None"
        )
        
        # Parse recommended offers
        df['recommended_offers_norm'] = df['Recommended_Offer_Pattern'].apply(normalize_offer_pattern)
        df['recommended_offers_str'] = df['recommended_offers_norm'].apply(
            lambda lst: ", ".join(lst) if lst else "None"
        )
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None": 66.34, "Voice": 2400},
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

# Cache functions for performance
@st.cache_data(persist=True)
def load_data(file_path):
    """Load and preprocess the data with caching"""
    try:
        # Try different file formats for optimization
        if file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        elif file_path.endswith('.feather'):
            df = pd.read_feather(file_path)
        else:  # CSV
            # For large CSV, use chunking
            df = pd.read_csv(file_path, low_memory=False)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None
    
    # Preprocess data
    df['MONTH'] = pd.to_numeric(df['MONTH'], errors='coerce').fillna(0).astype(int)
    df = df[df['MONTH'].between(1, 12)]
    
    # Parse offer patterns
    df['offer_pattern_norm'] = df['offer_pattern'].apply(normalize_offer_pattern)
    df['offer_pattern_str'] = df['offer_pattern_norm'].apply(
        lambda lst: ", ".join(lst) if lst else "None"
    )
    
    # Parse recommended offers
    df['recommended_offers_norm'] = df['Recommended_Offer_Pattern'].apply(normalize_offer_pattern)
    df['recommended_offers_str'] = df['recommended_offers_norm'].apply(
        lambda lst: ", ".join(lst) if lst else "None"
    )
    
    return df

def normalize_offer_pattern(val):
    """Convert raw stored offer pattern to a clean list of offer strings."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    
    if isinstance(val, str):
        s = val.strip()
        if not s or s.upper() in {"NONE", "NULL", "NAN", "NO_RECOMMENDATION"}:
            return []
        # Handle tuple/list strings
        if (s.startswith("(") and s.endswith(")")) or (s.startswith("[") and s.endswith("]")):
            try:
                lit = ast.literal_eval(s)
                if isinstance(lit, (list, tuple)):
                    return [str(x).strip() for x in lit if x and str(x).strip()]
                return [str(lit).strip()]
            except:
                pass
        # Handle comma-separated
        if "," in s:
            parts = [p.strip().strip("'\"") for p in s.split(",")]
            return [p for p in parts if p and p.upper() not in {"NONE", "NULL", "NAN"}]
        return [s]
    
    if isinstance(val, (list, tuple)):
        cleaned = []
        for x in val:
            if x is None or (isinstance(x, float) and pd.isna(x)):
                continue
            s = str(x).strip()
            if s and s.upper() not in {"NONE", "NULL", "NAN"}:
                cleaned.append(s)
        return cleaned
    
    return [str(val).strip()] if val else []

def check_password():
    """Returns `True` if the user had the correct password."""
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hashlib.sha256(st.session_state["password"].encode()).hexdigest() == hashlib.sha256(PASSWORD.encode()).hexdigest():
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # Create a centered login container
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("<br><br><br>", unsafe_allow_html=True)
            
            # Logo - centered
            logo_col1, logo_col2, logo_col3 = st.columns([1, 2, 1])
            with logo_col2:
                try:
                    logo = Image.open("touch.png")
                    st.image(logo, width=200, use_container_width=False)
                except:
                    st.markdown("## 📱 TOUCH", unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Styled login form
            st.markdown("""
                <div style='text-align: center; padding: 20px; background: #f0f2f6; border-radius: 10px;'>
                    <h3 style='color: #1A73E8;'>Access Portal</h3>
                    <p style='color: #5F6368;'>Please enter your password to continue</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Password input
            st.text_input(
                "Password", 
                type="password", 
                on_change=password_entered, 
                key="password",
                placeholder="Enter password..."
            )
            
            if "password_correct" in st.session_state and not st.session_state["password_correct"]:
                st.error("😕 Incorrect password. Please try again.")
            
            st.markdown("""
                <div style='text-align: center; margin-top: 50px; color: #999;'>
                    <small>Touch Recommendation Dashboard v1.0</small><br>
                    <small>© 2024 Touch Communications. All rights reserved.</small>
                </div>
            """, unsafe_allow_html=True)
        
        return False
    
    elif not st.session_state["password_correct"]:
        # Password not correct, show error
        col1, col2, col3 = st.columns([2, 2, 2])
        with col2:
            st.error("😕 Incorrect password. Please try again.")
            st.text_input(
                "Password", 
                type="password", 
                on_change=password_entered, 
                key="password"
            )
        return False
    else:
        # Password correct
        return True

def create_kpi_metrics(df, selected_month=None):
    """Calculate KPI metrics"""
    if selected_month:
        month_df = df[df['MONTH'] == selected_month]
    else:
        month_df = df
    
    metrics = {
        'total_customers': df['CustomerID'].nunique(),
        'total_arpu': df['ARPU'].sum(),
        'avg_arpu': df.groupby('CustomerID')['ARPU'].mean().mean(),
        'total_lift': df['Price_Difference'].sum(),
        'avg_lift': df['Price_Difference'].mean(),
        'positive_lift_pct': (df['Price_Difference'] > 0).mean() * 100,
        'month_arpu': month_df['ARPU'].sum() if selected_month else 0,
        'month_projected': (month_df['ARPU'] + month_df['Price_Difference']).sum() if selected_month else 0
    }
    
    return metrics

def create_arpu_chart(df, selected_month=None):
    """Create ARPU historical and projection charts"""
    # Historical ARPU by month
    hist_arpu = df.groupby('MONTH')['ARPU'].sum().reset_index()
    hist_arpu['Month_Name'] = hist_arpu['MONTH'].map(MONTH_ABBR)
    
    # Projected ARPU (current + lift)
    proj_arpu = df.groupby('MONTH').agg({
        'ARPU': 'sum',
        'Price_Difference': 'sum'
    }).reset_index()
    proj_arpu['Projected_ARPU'] = proj_arpu['ARPU'] + proj_arpu['Price_Difference']
    proj_arpu['Month_Name'] = proj_arpu['MONTH'].map(MONTH_ABBR)
    
    # Create subplot figure
    fig = go.Figure()
    
    # Add historical trace
    fig.add_trace(go.Scatter(
        x=hist_arpu['Month_Name'],
        y=hist_arpu['ARPU'],
        mode='lines+markers',
        name='Actual ARPU 2024',
        line=dict(color='#1A73E8', width=3),
        marker=dict(size=8)
    ))
    
    # Add projected trace
    fig.add_trace(go.Scatter(
        x=proj_arpu['Month_Name'],
        y=proj_arpu['Projected_ARPU'],
        mode='lines+markers',
        name='Projected ARPU 2025 (M-Series Transition)',
        line=dict(color='#25D366', width=3, dash='dash'),
        marker=dict(size=8)
    ))
    
    # Highlight selected month
    if selected_month:
        month_name = MONTH_ABBR[selected_month]
        month_idx = list(MONTH_ABBR.values()).index(month_name)
        
        # Add vertical band for selected month
        fig.add_vrect(
            x0=month_idx - 0.4, x1=month_idx + 0.4,
            fillcolor="#E0E0E0", opacity=0.3,
            layer="below", line_width=0
        )
    
    fig.update_layout(
        title="ARPU Trends: Historical vs Projected Impact",
        xaxis_title="Month",
        yaxis_title="ARPU ($)",
        template="plotly_white",
        height=400,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_persona_distribution(df, persona_type='Liquidity_Persona'):
    """Create distribution chart for personas"""
    dist = df.drop_duplicates('CustomerID')[persona_type].value_counts()
    
    fig = go.Figure(data=[
        go.Bar(
            x=dist.index,
            y=dist.values,
            marker_color='#1A73E8',
            text=dist.values,
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title=f"Customer Distribution by {persona_type.replace('_', ' ')}",
        xaxis_title=persona_type.replace('_', ' '),
        yaxis_title="Number of Customers",
        template="plotly_white",
        height=350,
        showlegend=False
    )
    
    return fig

def create_lift_distribution(df, liquidity=None, consumption=None):
    """Create ARPU lift distribution chart"""
    filtered_df = df.copy()
    
    if liquidity and liquidity != "All":
        filtered_df = filtered_df[filtered_df['Liquidity_Persona'] == liquidity]
    if consumption and consumption != "All":
        filtered_df = filtered_df[filtered_df['Consumption_Persona'] == consumption]
    
    # Group by month and calculate average lift
    lift_by_month = filtered_df.groupby('MONTH')['Price_Difference'].mean().reset_index()
    lift_by_month['Month_Name'] = lift_by_month['MONTH'].map(MONTH_ABBR)
    
    # Create color based on positive/negative
    colors = ['#25D366' if x > 0 else '#FF6B6B' for x in lift_by_month['Price_Difference']]
    
    fig = go.Figure(data=[
        go.Bar(
            x=lift_by_month['Month_Name'],
            y=lift_by_month['Price_Difference'],
            marker_color=colors,
            text=[f"${x:.2f}" for x in lift_by_month['Price_Difference']],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Average ARPU Lift by Month",
        xaxis_title="Month",
        yaxis_title="ARPU Lift ($)",
        template="plotly_white",
        height=350,
        showlegend=False
    )
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    return fig

def create_sub_persona_chart(df):
    """Create sub-persona distribution chart"""
    sub_dist = df.drop_duplicates('CustomerID')['Sub_Persona'].value_counts().head(10)
    
    fig = go.Figure(data=[
        go.Pie(
            labels=sub_dist.index,
            values=sub_dist.values,
            hole=0.4,
            marker=dict(
                colors=px.colors.qualitative.Set3[:len(sub_dist)]
            )
        )
    ])
    
    fig.update_layout(
        title="Customer Distribution by Sub-Persona",
        template="plotly_white",
        height=350,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05
        )
    )
    
    return fig

def create_device_distribution(df):
    """Create device category distribution"""
    device_dist = df.drop_duplicates('CustomerID')['Device_Category'].value_counts()
    
    # Calculate percentages
    total = device_dist.sum()
    percentages = (device_dist / total * 100).round(1)
    
    fig = go.Figure(data=[
        go.Bar(
            x=device_dist.index,
            y=device_dist.values,
            marker_color=['#1A73E8', '#4285F4', '#669DF6', '#AECBFA'][:len(device_dist)],
            text=[f"{device_dist[cat]}<br>({percentages[cat]}%)" for cat in device_dist.index],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Device Category Distribution",
        xaxis_title="Device Category",
        yaxis_title="Number of Customers",
        template="plotly_white",
        height=350,
        showlegend=False
    )
    
    return fig

def create_persona_matrix(df):
    """Create matrix showing customer counts by persona combination"""
    # Create pivot table of customer counts
    persona_counts = df.drop_duplicates('CustomerID').groupby(['Liquidity_Persona', 'Consumption_Persona']).size().reset_index(name='Count')
    persona_pivot = persona_counts.pivot(index='Liquidity_Persona', columns='Consumption_Persona', values='Count').fillna(0)
    
    fig = go.Figure(data=go.Heatmap(
        z=persona_pivot.values,
        x=persona_pivot.columns,
        y=persona_pivot.index,
        colorscale='Blues',
        text=persona_pivot.values.astype(int),
        texttemplate='%{text}',
        textfont={"size": 12},
        colorbar=dict(title="Customer Count")
    ))
    
    fig.update_layout(
        title="Customer Count by Persona Combination",
        xaxis_title="Consumption Persona",
        yaxis_title="Liquidity Persona",
        height=450,
        template="plotly_white"
    )
    
    return fig

def create_persona_lift_heatmap(df):
    """Create improved heatmap with better color scaling"""
    # Calculate average lift by personas
    persona_lift = df.groupby(['Liquidity_Persona', 'Consumption_Persona'])['Price_Difference'].mean().reset_index()
    persona_pivot = persona_lift.pivot(index='Liquidity_Persona', columns='Consumption_Persona', values='Price_Difference')
    
    # Use percentile-based color scaling to handle outliers
    values_flat = persona_pivot.values.flatten()
    values_flat = values_flat[~np.isnan(values_flat)]
    
    # Use log scale if there's a large range
    if len(values_flat) > 0:
        vmin = np.percentile(values_flat, 5)
        vmax = np.percentile(values_flat, 95)
    else:
        vmin, vmax = 0, 10
    
    fig = go.Figure(data=go.Heatmap(
        z=persona_pivot.values,
        x=persona_pivot.columns,
        y=persona_pivot.index,
        colorscale=[
            [0, '#FF4444'],      # Red for negative
            [0.25, '#FFB6B6'],   # Light red
            [0.5, '#FFFFFF'],    # White for zero
            [0.75, '#B6FFB6'],   # Light green
            [1, '#44FF44']       # Green for positive
        ],
        zmid=0,  # Center the color scale at 0
        zmin=vmin,
        zmax=vmax,
        text=np.round(persona_pivot.values, 2),
        texttemplate='$%{text}',
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

def calculate_persona_metrics(df, liquidity=None, consumption=None):
    """Calculate detailed metrics for persona deep dive"""
    filtered_df = df.copy()
    
    # Apply filters
    if liquidity and liquidity != "All":
        filtered_df = filtered_df[filtered_df['Liquidity_Persona'] == liquidity]
    if consumption and consumption != "All":
        filtered_df = filtered_df[filtered_df['Consumption_Persona'] == consumption]
    
    # Get unique customers for accurate counts
    unique_customers = filtered_df.drop_duplicates('CustomerID')
    
    # Calculate metrics
    metrics = {
        'total_customers': unique_customers.shape[0],
        'total_annual_mbs': filtered_df['MB_CONSUMPTION'].sum(),
        'avg_annual_mbs': filtered_df.groupby('CustomerID')['MB_CONSUMPTION'].sum().mean(),
        'total_voice': filtered_df['MINUTES'].sum(),
        'avg_voice': filtered_df.groupby('CustomerID')['MINUTES'].sum().mean(),
        'total_annual_spend': filtered_df['ARPU'].sum(),
        'avg_monthly_spend': filtered_df.groupby('CustomerID')['ARPU'].mean().mean(),
        'spend_volatility': filtered_df.groupby('CustomerID')['ARPU'].std().mean(),
        'arpu_lift_sum': filtered_df['Price_Difference'].sum(),
        'arpu_pct_of_total': 0,  # Will calculate after
        'arpu_lift_pct_of_total': 0,  # Will calculate after
    }
    
    # Calculate data to voice ratio
    total_data = filtered_df['MB_CONSUMPTION'].sum()
    total_voice = filtered_df['MINUTES'].sum()
    if total_voice > 0:
        metrics['total_data_to_voice'] = total_data / total_voice
        metrics['avg_data_to_voice'] = filtered_df.apply(
            lambda row: row['MB_CONSUMPTION'] / row['MINUTES'] if row['MINUTES'] > 0 else 0, axis=1
        ).mean()
    else:
        metrics['total_data_to_voice'] = 0
        metrics['avg_data_to_voice'] = 0
    
    # Device composition
    device_comp = unique_customers['Device_Category'].value_counts().to_dict()
    
    # Sub-persona composition
    sub_persona_comp = unique_customers['Sub_Persona'].value_counts().head(10).to_dict()
    
    return metrics, device_comp, sub_persona_comp

def main():
    # Check password first
    if not check_password():
        return
    
    # Header with logo
    col1, col2 = st.columns([1, 5])
    with col1:
        try:
            logo = Image.open("touch.png")
            st.image(logo, width=120)
        except:
            st.markdown("### 📱 TOUCH")
    
    with col2:
        st.title("RECOMMENDATION ENGINE DASHBOARD")
        st.markdown("*Personalized Offer Optimization Platform using 2-Way Clustering*")
    
    # Load data from Google Drive
    with st.spinner("Loading data..."):
        df = load_data()
    
    if df is None:
        st.error("Failed to load data. Please check the Google Drive link.")
        return
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    # Customer Type filter
    customer_type_options = ["All"] + sorted(df['Customer_Type'].dropna().unique().tolist())
    selected_customer_type = st.sidebar.selectbox(
        "Customer Type",
        customer_type_options,
        index=0
    )
    
    # Persona filters
    liquidity_options = ["All"] + sorted(df['Liquidity_Persona'].dropna().unique().tolist())
    selected_liquidity = st.sidebar.selectbox(
        "Liquidity Persona",
        liquidity_options,
        index=0
    )
    
    consumption_options = ["All"] + sorted(df['Consumption_Persona'].dropna().unique().tolist())
    selected_consumption = st.sidebar.selectbox(
        "Consumption Persona", 
        consumption_options,
        index=0
    )
    
    # Month slider
    selected_month = st.sidebar.slider(
        "Select Month",
        min_value=1,
        max_value=12,
        value=12,
        format="%d"
    )
    st.sidebar.markdown(f"**Selected:** {MONTH_NAMES[selected_month]}")
    
    # Apply filters
    filtered_df = df.copy()
    if selected_customer_type != "All":
        filtered_df = filtered_df[filtered_df['Customer_Type'] == selected_customer_type]
    if selected_liquidity != "All":
        filtered_df = filtered_df[filtered_df['Liquidity_Persona'] == selected_liquidity]
    if selected_consumption != "All":
        filtered_df = filtered_df[filtered_df['Consumption_Persona'] == selected_consumption]
    
    # Calculate metrics
    metrics = create_kpi_metrics(filtered_df, selected_month)
    
    # Calculate percentages for persona deep dive
    total_company_arpu = df['ARPU'].sum()
    total_company_lift = (df['ARPU'] + df['Price_Difference']).sum()
    
    # Main dashboard tabs - continue with rest of the UI code...
    """Convert raw stored offer pattern to a clean list of offer strings."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    
    if isinstance(val, str):
        s = val.strip()
        if not s or s.upper() in {"NONE", "NULL", "NAN", "NO_RECOMMENDATION"}:
            return []
        # Handle tuple/list strings
        if (s.startswith("(") and s.endswith(")")) or (s.startswith("[") and s.endswith("]")):
            try:
                lit = ast.literal_eval(s)
                if isinstance(lit, (list, tuple)):
                    return [str(x).strip() for x in lit if x and str(x).strip()]
                return [str(lit).strip()]
            except:
                pass
        # Handle comma-separated
        if "," in s:
            parts = [p.strip().strip("'\"") for p in s.split(",")]
            return [p for p in parts if p and p.upper() not in {"NONE", "NULL", "NAN"}]
        return [s]
    
    if isinstance(val, (list, tuple)):
        cleaned = []
        for x in val:
            if x is None or (isinstance(x, float) and pd.isna(x)):
                continue
            s = str(x).strip()
            if s and s.upper() not in {"NONE", "NULL", "NAN"}:
                cleaned.append(s)
        return cleaned
    
    return [str(val).strip()] if val else []

def check_password():
    """Returns `True` if the user had the correct password."""
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hashlib.sha256(st.session_state["password"].encode()).hexdigest() == hashlib.sha256(PASSWORD.encode()).hexdigest():
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # Create a centered login container
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("<br><br><br>", unsafe_allow_html=True)
            
            # Logo - centered
            logo_col1, logo_col2, logo_col3 = st.columns([1, 2, 1])
            with logo_col2:
                try:
                    logo = Image.open("touch.png")
                    st.image(logo, width=200, use_container_width=False)
                except:
                    st.markdown("## 📱 TOUCH", unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Styled login form
            st.markdown("""
                <div style='text-align: center; padding: 20px; background: #f0f2f6; border-radius: 10px;'>
                    <h3 style='color: #1A73E8;'>Access Portal</h3>
                    <p style='color: #5F6368;'>Please enter your password to continue</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Password input
            st.text_input(
                "Password", 
                type="password", 
                on_change=password_entered, 
                key="password",
                placeholder="Enter password..."
            )
            
            if "password_correct" in st.session_state and not st.session_state["password_correct"]:
                st.error("😕 Incorrect password. Please try again.")
            
            st.markdown("""
                <div style='text-align: center; margin-top: 50px; color: #999;'>
                    <small>Touch Recommendation Dashboard v1.0</small><br>
                    <small>© 2024 Touch Communications. All rights reserved.</small>
                </div>
            """, unsafe_allow_html=True)
        
        return False
    
    elif not st.session_state["password_correct"]:
        # Password not correct, show error
        col1, col2, col3 = st.columns([2, 2, 2])
        with col2:
            st.error("😕 Incorrect password. Please try again.")
            st.text_input(
                "Password", 
                type="password", 
                on_change=password_entered, 
                key="password"
            )
        return False
    else:
        # Password correct
        return True

def create_kpi_metrics(df, selected_month=None):
    """Calculate KPI metrics"""
    if selected_month:
        month_df = df[df['MONTH'] == selected_month]
    else:
        month_df = df
    
    metrics = {
        'total_customers': df['CustomerID'].nunique(),
        'total_arpu': df['ARPU'].sum(),
        'avg_arpu': df.groupby('CustomerID')['ARPU'].mean().mean(),
        'total_lift': df['Price_Difference'].sum(),
        'avg_lift': df['Price_Difference'].mean(),
        'positive_lift_pct': (df['Price_Difference'] > 0).mean() * 100,
        'month_arpu': month_df['ARPU'].sum() if selected_month else 0,
        'month_projected': (month_df['ARPU'] + month_df['Price_Difference']).sum() if selected_month else 0
    }
    
    return metrics

def create_arpu_chart(df, selected_month=None):
    """Create ARPU historical and projection charts"""
    # Historical ARPU by month
    hist_arpu = df.groupby('MONTH')['ARPU'].sum().reset_index()
    hist_arpu['Month_Name'] = hist_arpu['MONTH'].map(MONTH_ABBR)
    
    # Projected ARPU (current + lift)
    proj_arpu = df.groupby('MONTH').agg({
        'ARPU': 'sum',
        'Price_Difference': 'sum'
    }).reset_index()
    proj_arpu['Projected_ARPU'] = proj_arpu['ARPU'] + proj_arpu['Price_Difference']
    proj_arpu['Month_Name'] = proj_arpu['MONTH'].map(MONTH_ABBR)
    
    # Create subplot figure
    fig = go.Figure()
    
    # Add historical trace
    fig.add_trace(go.Scatter(
        x=hist_arpu['Month_Name'],
        y=hist_arpu['ARPU'],
        mode='lines+markers',
        name='Actual ARPU 2024',
        line=dict(color='#1A73E8', width=3),
        marker=dict(size=8)
    ))
    
    # Add projected trace
    fig.add_trace(go.Scatter(
        x=proj_arpu['Month_Name'],
        y=proj_arpu['Projected_ARPU'],
        mode='lines+markers',
        name='Projected ARPU 2025 (M-Series Transition)',
        line=dict(color='#25D366', width=3, dash='dash'),
        marker=dict(size=8)
    ))
    
    # Highlight selected month
    if selected_month:
        month_name = MONTH_ABBR[selected_month]
        month_idx = list(MONTH_ABBR.values()).index(month_name)
        
        # Add vertical band for selected month
        fig.add_vrect(
            x0=month_idx - 0.4, x1=month_idx + 0.4,
            fillcolor="#E0E0E0", opacity=0.3,
            layer="below", line_width=0
        )
    
    fig.update_layout(
        title="ARPU Trends: Historical vs Projected Impact",
        xaxis_title="Month",
        yaxis_title="ARPU ($)",
        template="plotly_white",
        height=400,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_persona_distribution(df, persona_type='Liquidity_Persona'):
    """Create distribution chart for personas"""
    dist = df.drop_duplicates('CustomerID')[persona_type].value_counts()
    
    fig = go.Figure(data=[
        go.Bar(
            x=dist.index,
            y=dist.values,
            marker_color='#1A73E8',
            text=dist.values,
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title=f"Customer Distribution by {persona_type.replace('_', ' ')}",
        xaxis_title=persona_type.replace('_', ' '),
        yaxis_title="Number of Customers",
        template="plotly_white",
        height=350,
        showlegend=False
    )
    
    return fig

def create_lift_distribution(df, liquidity=None, consumption=None):
    """Create ARPU lift distribution chart"""
    filtered_df = df.copy()
    
    if liquidity and liquidity != "All":
        filtered_df = filtered_df[filtered_df['Liquidity_Persona'] == liquidity]
    if consumption and consumption != "All":
        filtered_df = filtered_df[filtered_df['Consumption_Persona'] == consumption]
    
    # Group by month and calculate average lift
    lift_by_month = filtered_df.groupby('MONTH')['Price_Difference'].mean().reset_index()
    lift_by_month['Month_Name'] = lift_by_month['MONTH'].map(MONTH_ABBR)
    
    # Create color based on positive/negative
    colors = ['#25D366' if x > 0 else '#FF6B6B' for x in lift_by_month['Price_Difference']]
    
    fig = go.Figure(data=[
        go.Bar(
            x=lift_by_month['Month_Name'],
            y=lift_by_month['Price_Difference'],
            marker_color=colors,
            text=[f"${x:.2f}" for x in lift_by_month['Price_Difference']],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Average ARPU Lift by Month",
        xaxis_title="Month",
        yaxis_title="ARPU Lift ($)",
        template="plotly_white",
        height=350,
        showlegend=False
    )
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    return fig

def create_sub_persona_chart(df):
    """Create sub-persona distribution chart"""
    sub_dist = df.drop_duplicates('CustomerID')['Sub_Persona'].value_counts().head(10)
    
    fig = go.Figure(data=[
        go.Pie(
            labels=sub_dist.index,
            values=sub_dist.values,
            hole=0.4,
            marker=dict(
                colors=px.colors.qualitative.Set3[:len(sub_dist)]
            )
        )
    ])
    
    fig.update_layout(
        title="Customer Distribution by Sub-Persona",
        template="plotly_white",
        height=350,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05
        )
    )
    
    return fig

def create_device_distribution(df):
    """Create device category distribution"""
    device_dist = df.drop_duplicates('CustomerID')['Device_Category'].value_counts()
    
    # Calculate percentages
    total = device_dist.sum()
    percentages = (device_dist / total * 100).round(1)
    
    fig = go.Figure(data=[
        go.Bar(
            x=device_dist.index,
            y=device_dist.values,
            marker_color=['#1A73E8', '#4285F4', '#669DF6', '#AECBFA'][:len(device_dist)],
            text=[f"{device_dist[cat]}<br>({percentages[cat]}%)" for cat in device_dist.index],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Device Category Distribution",
        xaxis_title="Device Category",
        yaxis_title="Number of Customers",
        template="plotly_white",
        height=350,
        showlegend=False
    )
    
    return fig

def create_persona_matrix(df):
    """Create matrix showing customer counts by persona combination"""
    # Create pivot table of customer counts
    persona_counts = df.drop_duplicates('CustomerID').groupby(['Liquidity_Persona', 'Consumption_Persona']).size().reset_index(name='Count')
    persona_pivot = persona_counts.pivot(index='Liquidity_Persona', columns='Consumption_Persona', values='Count').fillna(0)
    
    fig = go.Figure(data=go.Heatmap(
        z=persona_pivot.values,
        x=persona_pivot.columns,
        y=persona_pivot.index,
        colorscale='Blues',
        text=persona_pivot.values.astype(int),
        texttemplate='%{text}',
        textfont={"size": 12},
        colorbar=dict(title="Customer Count")
    ))
    
    fig.update_layout(
        title="Customer Count by Persona Combination",
        xaxis_title="Consumption Persona",
        yaxis_title="Liquidity Persona",
        height=450,
        template="plotly_white"
    )
    
    return fig

def create_persona_lift_heatmap(df):
    """Create improved heatmap with better color scaling"""
    # Calculate average lift by personas
    persona_lift = df.groupby(['Liquidity_Persona', 'Consumption_Persona'])['Price_Difference'].mean().reset_index()
    persona_pivot = persona_lift.pivot(index='Liquidity_Persona', columns='Consumption_Persona', values='Price_Difference')
    
    # Use percentile-based color scaling to handle outliers
    values_flat = persona_pivot.values.flatten()
    values_flat = values_flat[~np.isnan(values_flat)]
    
    # Use log scale if there's a large range
    if len(values_flat) > 0:
        vmin = np.percentile(values_flat, 5)
        vmax = np.percentile(values_flat, 95)
    else:
        vmin, vmax = 0, 10
    
    fig = go.Figure(data=go.Heatmap(
        z=persona_pivot.values,
        x=persona_pivot.columns,
        y=persona_pivot.index,
        colorscale=[
            [0, '#FF4444'],      # Red for negative
            [0.25, '#FFB6B6'],   # Light red
            [0.5, '#FFFFFF'],    # White for zero
            [0.75, '#B6FFB6'],   # Light green
            [1, '#44FF44']       # Green for positive
        ],
        zmid=0,  # Center the color scale at 0
        zmin=vmin,
        zmax=vmax,
        text=np.round(persona_pivot.values, 2),
        texttemplate='$%{text}',
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

def calculate_persona_metrics(df, liquidity=None, consumption=None):
    """Calculate detailed metrics for persona deep dive"""
    filtered_df = df.copy()
    
    # Apply filters
    if liquidity and liquidity != "All":
        filtered_df = filtered_df[filtered_df['Liquidity_Persona'] == liquidity]
    if consumption and consumption != "All":
        filtered_df = filtered_df[filtered_df['Consumption_Persona'] == consumption]
    
    # Get unique customers for accurate counts
    unique_customers = filtered_df.drop_duplicates('CustomerID')
    
    # Calculate metrics
    metrics = {
        'total_customers': unique_customers.shape[0],
        'total_annual_mbs': filtered_df['MB_CONSUMPTION'].sum(),
        'avg_annual_mbs': filtered_df.groupby('CustomerID')['MB_CONSUMPTION'].sum().mean(),
        'total_voice': filtered_df['MINUTES'].sum(),
        'avg_voice': filtered_df.groupby('CustomerID')['MINUTES'].sum().mean(),
        'total_annual_spend': filtered_df['ARPU'].sum(),
        'avg_monthly_spend': filtered_df.groupby('CustomerID')['ARPU'].mean().mean(),
        'spend_volatility': filtered_df.groupby('CustomerID')['ARPU'].std().mean(),
        'arpu_lift_sum': filtered_df['Price_Difference'].sum(),
        'arpu_pct_of_total': 0,  # Will calculate after
        'arpu_lift_pct_of_total': 0,  # Will calculate after
    }
    
    # Calculate data to voice ratio
    total_data = filtered_df['MB_CONSUMPTION'].sum()
    total_voice = filtered_df['MINUTES'].sum()
    if total_voice > 0:
        metrics['total_data_to_voice'] = total_data / total_voice
        metrics['avg_data_to_voice'] = filtered_df.apply(
            lambda row: row['MB_CONSUMPTION'] / row['MINUTES'] if row['MINUTES'] > 0 else 0, axis=1
        ).mean()
    else:
        metrics['total_data_to_voice'] = 0
        metrics['avg_data_to_voice'] = 0
    
    # Device composition
    device_comp = unique_customers['Device_Category'].value_counts().to_dict()
    
    # Sub-persona composition
    sub_persona_comp = unique_customers['Sub_Persona'].value_counts().head(10).to_dict()
    
    return metrics, device_comp, sub_persona_comp

def main():
    # Check password first
    if not check_password():
        return
    
    # Header with logo
    col1, col2 = st.columns([1, 5])
    with col1:
        try:
            logo = Image.open("touch.png")
            st.image(logo, width=120)
        except:
            st.markdown("### 📱 TOUCH")
    
    with col2:
        st.title("RECOMMENDATION ENGINE DASHBOARD")
        st.markdown("*Personalized Offer Optimization Platform using 2-Way Clustering*")
    
    # Load data
    DATA_PATH = r"C:\Users\omarr\OneDrive\Desktop\touch_dashboard\data.parquet"
    
    with st.spinner("Loading data... This may take a moment for large files."):
        df = load_data(DATA_PATH)
    
    if df is None:
        st.error("Failed to load data. Please check the file path.")
        return
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    # Customer Type filter (NEW)
    customer_type_options = ["All"] + sorted(df['Customer_Type'].dropna().unique().tolist())
    selected_customer_type = st.sidebar.selectbox(
        "Customer Type",
        customer_type_options,
        index=0
    )
    
    # Persona filters
    liquidity_options = ["All"] + sorted(df['Liquidity_Persona'].dropna().unique().tolist())
    selected_liquidity = st.sidebar.selectbox(
        "Liquidity Persona",
        liquidity_options,
        index=0
    )
    
    consumption_options = ["All"] + sorted(df['Consumption_Persona'].dropna().unique().tolist())
    selected_consumption = st.sidebar.selectbox(
        "Consumption Persona", 
        consumption_options,
        index=0
    )
    
    # Month slider
    selected_month = st.sidebar.slider(
        "Select Month",
        min_value=1,
        max_value=12,
        value=12,
        format="%d"
    )
    st.sidebar.markdown(f"**Selected:** {MONTH_NAMES[selected_month]}")
    
    # Apply filters
    filtered_df = df.copy()
    if selected_customer_type != "All":
        filtered_df = filtered_df[filtered_df['Customer_Type'] == selected_customer_type]
    if selected_liquidity != "All":
        filtered_df = filtered_df[filtered_df['Liquidity_Persona'] == selected_liquidity]
    if selected_consumption != "All":
        filtered_df = filtered_df[filtered_df['Consumption_Persona'] == selected_consumption]
    
    # Calculate metrics
    metrics = create_kpi_metrics(filtered_df, selected_month)
    
    # Calculate percentages for persona deep dive
    total_company_arpu = df['ARPU'].sum()
    total_company_lift = (df['ARPU'] + df['Price_Difference']).sum()
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Recommendation Overview", 
        "🔍 Customer Explorer", 
        "📈 Analytics", 
        "🎯 Persona Deep Dive",
        "📋 Offer Catalog"
    ])
    
    # Tab 1: Recommendation Overview
    with tab1:
        # KPI Metrics Row 1
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Total Customers",
                f"{metrics['total_customers']:,}",
                delta=None
            )
        with col2:
            st.metric(
                "Total Annual ARPU",
                f"${metrics['total_arpu']:,.0f}",
                delta=None
            )
        with col3:
            st.metric(
                "Projected Total ARPU Lift",
                f"${metrics['total_lift']:,.0f}",
                delta=f"{metrics['positive_lift_pct']:.1f}% positive",
                delta_color="normal"
            )
        with col4:
            st.metric(
                "Avg ARPU Lift per Customer",
                f"${metrics['avg_lift']:.2f}",
                delta=None
            )
        
        # KPI Metrics Row 2 - Month specific
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                f"ARPU - {MONTH_ABBR[selected_month]}",
                f"${metrics['month_arpu']:,.0f}",
                delta=None
            )
        with col2:
            st.metric(
                f"Projected - {MONTH_ABBR[selected_month]}",
                f"${metrics['month_projected']:,.0f}",
                delta=f"+${metrics['month_projected'] - metrics['month_arpu']:,.0f}",
                delta_color="normal"
            )
        with col3:
            month_df = filtered_df[filtered_df['MONTH'] == selected_month]
            active_customers = month_df['CustomerID'].nunique()
            st.metric(
                f"Active Customers - {MONTH_ABBR[selected_month]}",
                f"{active_customers:,}",
                delta=None
            )
        with col4:
            avg_month_lift = month_df['Price_Difference'].mean() if not month_df.empty else 0
            st.metric(
                f"Avg Lift - {MONTH_ABBR[selected_month]}",
                f"${avg_month_lift:.2f}",
                delta=None
            )
        
        # Charts Row 1
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(
                create_arpu_chart(filtered_df, selected_month),
                use_container_width=True
            )
        
        with col2:
            st.plotly_chart(
                create_lift_distribution(filtered_df, selected_liquidity, selected_consumption),
                use_container_width=True
            )
        
        # Charts Row 2
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(
                create_persona_matrix(filtered_df),
                use_container_width=True
            )
        
        with col2:
            st.plotly_chart(
                create_persona_lift_heatmap(filtered_df),
                use_container_width=True
            )
        
        # Charts Row 3
        col1, col2 = st.columns(2)
        
        with col1:
            persona_type = st.selectbox(
                "Select Persona View",
                ["Liquidity_Persona", "Consumption_Persona"],
                label_visibility="collapsed"
            )
            st.plotly_chart(
                create_persona_distribution(filtered_df, persona_type),
                use_container_width=True
            )
        
        with col2:
            st.plotly_chart(
                create_device_distribution(filtered_df),
                use_container_width=True
            )
        
        # Charts Row 4
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(
                create_sub_persona_chart(filtered_df),
                use_container_width=True
            )
        
        with col2:
            # Additional insights
            st.markdown("### 📊 Quick Insights")
            
            # Calculate some insights
            top_sub_persona = filtered_df.drop_duplicates('CustomerID')['Sub_Persona'].value_counts().index[0]
            avg_consumption = filtered_df['MB_CONSUMPTION'].mean()
            top_device = filtered_df.drop_duplicates('CustomerID')['Device_Category'].value_counts().index[0]
            
            st.info(f"""
            **Key Findings:**
            - Most common sub-persona: **{top_sub_persona}**
            - Average data consumption: **{avg_consumption:,.0f} MB**
            - Dominant device category: **{top_device}**
            - Customers with positive lift: **{metrics['positive_lift_pct']:.1f}%**
            - Total projected ARPU increase: **${metrics['total_lift']:,.0f}**
            """)
    
    # Tab 2: Customer Explorer
    with tab2:
        st.markdown("### 🔍 Customer Explorer")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Search and filters
            st.markdown("#### Search & Filter")
            
            # Customer search
            search_term = st.text_input("Search Customer ID", placeholder="Enter Customer ID...")
            
            # Customer Type filter for explorer (NEW)
            explorer_customer_type = st.selectbox(
                "Customer Type",
                ["All"] + sorted(filtered_df['Customer_Type'].dropna().unique().tolist()),
                key="explorer_customer_type"
            )
            
            # Device Category filter
            explorer_device = st.selectbox(
                "Device Category",
                ["All"] + sorted(filtered_df['Device_Category'].dropna().unique().tolist())
            )
            
            # Apply explorer filters
            explorer_df = filtered_df.copy()
            if explorer_customer_type != "All":
                explorer_df = explorer_df[explorer_df['Customer_Type'] == explorer_customer_type]
            if explorer_device != "All":
                explorer_df = explorer_df[explorer_df['Device_Category'] == explorer_device]
            
            # Get all unique customer IDs
            all_customer_ids = sorted(explorer_df['CustomerID'].dropna().unique())
            total_customers = len(all_customer_ids)
            
            # Filter by search term
            if search_term:
                # Filter customer IDs that contain the search term
                filtered_customer_ids = [cid for cid in all_customer_ids 
                                        if search_term.lower() in str(cid).lower()]
            else:
                filtered_customer_ids = all_customer_ids
            
            # Customer list with pagination
            st.markdown("#### Customer List")
            
            # Pagination controls
            items_per_page = 50
            total_filtered = len(filtered_customer_ids)
            total_pages = max(1, (total_filtered + items_per_page - 1) // items_per_page)
            
            # Create pagination controls in columns
            page_col1, page_col2, page_col3 = st.columns([1, 2, 1])
            with page_col2:
                if total_pages > 1:
                    current_page = st.number_input(
                        "Page", 
                        min_value=1, 
                        max_value=total_pages, 
                        value=1, 
                        step=1,
                        label_visibility="collapsed"
                    )
                else:
                    current_page = 1
            
            # Calculate page boundaries
            start_idx = (current_page - 1) * items_per_page
            end_idx = min(start_idx + items_per_page, total_filtered)
            
            # Get customers for current page
            page_customer_ids = filtered_customer_ids[start_idx:end_idx]
            
            # Customer selector
            if page_customer_ids:
                selected_customer = st.selectbox(
                    "Select Customer",
                    page_customer_ids,
                    label_visibility="collapsed",
                    key=f"customer_select_{current_page}"
                )
            else:
                selected_customer = None
                st.warning("No customers found matching the search criteria")
            
            # Info about current view
            if search_term:
                st.info(f"Found {total_filtered:,} customers matching '{search_term}'")
                if total_pages > 1:
                    st.info(f"Page {current_page} of {total_pages} | Showing {start_idx+1}-{end_idx} of {total_filtered:,}")
            else:
                st.info(f"Total: {total_customers:,} customers")
                if total_pages > 1:
                    st.info(f"Page {current_page} of {total_pages} | Showing {start_idx+1}-{end_idx}")
        
        with col2:
            if selected_customer:
                # Get customer data for selected month
                cust_data = explorer_df[
                    (explorer_df['CustomerID'] == selected_customer) & 
                    (explorer_df['MONTH'] == selected_month)
                ]
                
                if not cust_data.empty:
                    row = cust_data.iloc[0]
                    
                    # Customer details
                    st.markdown(f"### Customer: {selected_customer}")
                    
                    col_a, col_b, col_c, col_d = st.columns(4)
                    with col_a:
                        st.markdown(f"**Type:** {row.get('Customer_Type', 'N/A')}")
                    with col_b:
                        st.markdown(f"**Liquidity:** {row.get('Liquidity_Persona', 'N/A')}")
                    with col_c:
                        st.markdown(f"**Consumption:** {row.get('Consumption_Persona', 'N/A')}")
                    with col_d:
                        st.markdown(f"**Sub-Persona:** {row.get('Sub_Persona', 'N/A')}")
                    
                    st.markdown("---")
                    
                    # Current Status
                    st.markdown(f"#### Current Status - {MONTH_NAMES[selected_month]}")
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        current_offers = row.get('offer_pattern_str', 'None')
                        st.markdown(f"**Current Offers:** {current_offers}")
                        
                        mb_consumption = row.get('MB_CONSUMPTION', 0)
                        mb_allowance = row.get('mb_allowance', 0)
                        mb_usage_pct = row.get('mb_usage_pct', 0)
                        st.markdown(f"**Data Usage:** {mb_consumption:,.0f} MB / {mb_allowance:,.0f} MB ({mb_usage_pct:.1f}%)")
                    
                    with col_b:
                        arpu = row.get('ARPU', 0)
                        st.markdown(f"**Current ARPU:** ${arpu:.2f}")
                        
                        minutes = row.get('MINUTES', 0)
                        st.markdown(f"**Voice Minutes:** {minutes:,.0f}")
                    
                    st.markdown("---")
                    
                    # Recommendation
                    st.markdown("#### Recommendation")
                    
                    recommended = row.get('recommended_offers_str', 'None')
                    if recommended and recommended != "None" and recommended != "NO_RECOMMENDATION":
                        st.success(f"**Recommended Plan:** {recommended}")
                    else:
                        st.warning("**No recommendation available**")
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        price_diff = row.get('Price_Difference', 0)
                        color = "green" if price_diff > 0 else "red" if price_diff < 0 else "gray"
                        st.markdown(f"**ARPU Lift:** <span style='color:{color}; font-weight:bold;'>${price_diff:.2f}</span>", 
                                  unsafe_allow_html=True)
                    
                    with col_b:
                        new_price = row.get('Recommended_Offer_Price', 0)
                        st.markdown(f"**New Price:** ${new_price:.2f}")
                    
                    # Recommendation message
                    message = row.get('Message_English', '')
                    if message:
                        st.info(f"📱 {message}")
                    
                    # Historical view
                    st.markdown("---")
                    st.markdown("#### Customer History (All Months)")
                    
                    # Get all months data for this customer
                    cust_history = explorer_df[explorer_df['CustomerID'] == selected_customer].sort_values('MONTH')
                    
                    # Create mini chart
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=cust_history['MONTH'].map(MONTH_ABBR),
                        y=cust_history['ARPU'],
                        mode='lines+markers',
                        name='Actual ARPU',
                        line=dict(color='#1A73E8', width=2)
                    ))
                    fig.add_trace(go.Scatter(
                        x=cust_history['MONTH'].map(MONTH_ABBR),
                        y=cust_history['ARPU'] + cust_history['Price_Difference'],
                        mode='lines+markers',
                        name='Potential ARPU',
                        line=dict(color='#25D366', width=2, dash='dash')
                    ))
                    fig.update_layout(
                        title="Customer ARPU Trend",
                        height=300,
                        template="plotly_white",
                        showlegend=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    st.warning(f"No data available for customer {selected_customer} in {MONTH_NAMES[selected_month]}")
    
    # Tab 3: Analytics
    with tab3:
        st.markdown("### 📈 Advanced Analytics")
        
        # Offer migration analysis
        st.markdown("#### Offer Migration Analysis")
        
        # Filter out 'None' and 'NO_RECOMMENDATION' for cleaner analysis
        current_offers_filtered = filtered_df[
            ~filtered_df['offer_pattern_str'].isin(['None', 'NO_RECOMMENDATION'])
        ]['offer_pattern_str'].value_counts().head(10)
        
        recommended_offers_filtered = filtered_df[
            ~filtered_df['recommended_offers_str'].isin(['None', 'NO_RECOMMENDATION'])
        ]['recommended_offers_str'].value_counts().head(10)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Top Current Offers**")
            if not current_offers_filtered.empty:
                fig = go.Figure(data=[
                    go.Bar(
                        y=current_offers_filtered.index,
                        x=current_offers_filtered.values,
                        orientation='h',
                        marker_color='#4285F4'
                    )
                ])
                fig.update_layout(
                    height=400,
                    template="plotly_white",
                    xaxis_title="Number of Customers",
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No current offers to display")
        
        with col2:
            st.markdown("**Top Recommended Offers**")
            if not recommended_offers_filtered.empty:
                fig = go.Figure(data=[
                    go.Bar(
                        y=recommended_offers_filtered.index,
                        x=recommended_offers_filtered.values,
                        orientation='h',
                        marker_color='#25D366'
                    )
                ])
                fig.update_layout(
                    height=400,
                    template="plotly_white",
                    xaxis_title="Number of Customers",
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No recommended offers to display")
        
        # Summary statistics
        st.markdown("---")
        st.markdown("#### Summary Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Data Coverage**")
            total_months = filtered_df['MONTH'].nunique()
            total_records = len(filtered_df)
            avg_records_per_customer = filtered_df.groupby('CustomerID').size().mean()
            
            st.metric("Total Months", total_months)
            st.metric("Total Records", f"{total_records:,}")
            st.metric("Avg Records/Customer", f"{avg_records_per_customer:.1f}")
        
        with col2:
            st.markdown("**Recommendation Impact**")
            no_rec_count = (filtered_df['recommended_offers_str'] == 'NO_RECOMMENDATION').sum()
            rec_rate = ((len(filtered_df) - no_rec_count) / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
            positive_lift = (filtered_df['Price_Difference'] > 0).sum()
            
            st.metric("Recommendation Rate", f"{rec_rate:.1f}%")
            st.metric("Positive Lift Cases", f"{positive_lift:,}")
            st.metric("No Recommendation", f"{no_rec_count:,}")
        
        with col3:
            st.markdown("**Financial Impact**")
            total_current = filtered_df['ARPU'].sum()
            total_recommended = (filtered_df['ARPU'] + filtered_df['Price_Difference']).sum()
            growth_rate = ((total_recommended - total_current) / total_current * 100) if total_current > 0 else 0
            
            st.metric("Current Total ARPU", f"${total_current:,.0f}")
            st.metric("Projected Total ARPU", f"${total_recommended:,.0f}")
            st.metric("Growth Rate", f"{growth_rate:.2f}%")
    
    # Tab 4: Persona Deep Dive (NEW)
    with tab4:
        st.markdown("### 🎯 Persona Deep Dive")
        st.markdown("*Change the filters to explore different persona combinations*")
        
        # Calculate metrics for selected personas
        persona_metrics, device_comp, sub_persona_comp = calculate_persona_metrics(
            filtered_df, selected_liquidity, selected_consumption
        )
        
        # Calculate percentages
        persona_metrics['arpu_pct_of_total'] = (persona_metrics['total_annual_spend'] / total_company_arpu * 100) if total_company_arpu > 0 else 0
        persona_metrics['arpu_lift_pct_of_total'] = ((persona_metrics['total_annual_spend'] + persona_metrics['arpu_lift_sum']) / total_company_lift * 100) if total_company_lift > 0 else 0
        
        # Display metrics in columns
        st.markdown("#### 📊 Key Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Customers", f"{persona_metrics['total_customers']:,}")
            st.metric("Total Annual MBs", f"{persona_metrics['total_annual_mbs']:,.0f}")
            st.metric("Avg Annual MBs", f"{persona_metrics['avg_annual_mbs']:,.0f}")
        
        with col2:
            st.metric("Total Voice Minutes", f"{persona_metrics['total_voice']:,.0f}")
            st.metric("Avg Voice Minutes", f"{persona_metrics['avg_voice']:,.0f}")
            st.metric("Data/Voice Ratio", f"{persona_metrics['avg_data_to_voice']:.2f}")
        
        with col3:
            st.metric("Total Annual Spend", f"${persona_metrics['total_annual_spend']:,.0f}")
            st.metric("Avg Monthly Spend", f"${persona_metrics['avg_monthly_spend']:.2f}")
            st.metric("Spend Volatility", f"${persona_metrics['spend_volatility']:.2f}")
        
        with col4:
            st.metric("ARPU Lift Sum", f"${persona_metrics['arpu_lift_sum']:,.0f}")
            st.metric("% of Total ARPU", f"{persona_metrics['arpu_pct_of_total']:.1f}%")
            st.metric("% of Projected ARPU", f"{persona_metrics['arpu_lift_pct_of_total']:.1f}%")
        
        st.markdown("---")
        
        # Device and Sub-persona composition
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📱 Device Composition")
            if device_comp:
                device_df = pd.DataFrame(list(device_comp.items()), columns=['Device', 'Count'])
                device_df['Percentage'] = (device_df['Count'] / device_df['Count'].sum() * 100).round(1)
                
                fig = go.Figure(data=[
                    go.Pie(
                        labels=device_df['Device'],
                        values=device_df['Count'],
                        hole=0.4,
                        textinfo='label+percent',
                        marker=dict(colors=px.colors.qualitative.Set3[:len(device_df)])
                    )
                ])
                fig.update_layout(
                    height=350,
                    showlegend=True,
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No device data available")
        
        with col2:
            st.markdown("#### 👥 Sub-Persona Composition")
            if sub_persona_comp:
                sub_df = pd.DataFrame(list(sub_persona_comp.items()), columns=['Sub-Persona', 'Count'])
                sub_df = sub_df.sort_values('Count', ascending=True)
                
                fig = go.Figure(data=[
                    go.Bar(
                        y=sub_df['Sub-Persona'],
                        x=sub_df['Count'],
                        orientation='h',
                        marker_color='#1A73E8',
                        text=sub_df['Count'],
                        textposition='auto'
                    )
                ])
                fig.update_layout(
                    height=350,
                    xaxis_title="Number of Customers",
                    template="plotly_white",
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No sub-persona data available")
        
        # Additional insights for persona
        st.markdown("---")
        st.markdown("#### 💡 Persona Insights")
        
        # Show filters applied
        filters_text = []
        if selected_customer_type != "All":
            filters_text.append(f"Customer Type: **{selected_customer_type}**")
        if selected_liquidity != "All":
            filters_text.append(f"Liquidity: **{selected_liquidity}**")
        if selected_consumption != "All":
            filters_text.append(f"Consumption: **{selected_consumption}**")
        
        if filters_text:
            st.info("Current filters: " + " | ".join(filters_text))
        else:
            st.info("Showing data for **All Personas**")
        
        # Key characteristics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Persona Insights")
            st.write(f"""
            - **Avg Data Usage:** {persona_metrics['avg_annual_mbs']/12:.0f} MB/month
            - **Avg Voice Usage:** {persona_metrics['avg_voice']/12:.0f} minutes/month
            """)
        
        with col2:
            st.markdown("##### Business Impact")
            st.write(f"""
            - **Revenue Contribution:** {persona_metrics['arpu_pct_of_total']:.1f}% of total ARPU
            - **Growth Potential:** ${persona_metrics['arpu_lift_sum']:,.0f} total lift opportunity
            - **Lift per Customer:** ${persona_metrics['arpu_lift_sum']/persona_metrics['total_customers']:.2f} average
            - **Customer Base:** {persona_metrics['total_customers']:,} customers
            - **Post-Optimization Share:** {persona_metrics['arpu_lift_pct_of_total']:.1f}% of projected ARPU
            """)
    
    # Tab 5: Offer Catalog
    with tab5:
        st.markdown("### 📋 Offer Catalog")
        st.markdown("Browse all available offers organized by category")
        
        # Create tabs for each offer category
        offer_tabs = st.tabs(list(OFFER_CATALOG.keys()))
        
        for i, (category, df_offers) in enumerate(OFFER_CATALOG.items()):
            with offer_tabs[i]:
                st.markdown(f"#### {category}")
                
                # Format the dataframe for display
                display_df = df_offers.copy()
                display_df['Data (GB)'] = (display_df['Data (MB)'] / 1024).round(2)
                display_df = display_df[['Code', 'Name', 'Type', 'Data (MB)', 'Data (GB)', 'Price ($)', 'Voice']]
                
                # Apply styling
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Code": st.column_config.TextColumn("Offer Code", width="medium"),
                        "Name": st.column_config.TextColumn("Offer Name", width="medium"),
                        "Type": st.column_config.TextColumn("Type", width="small"),
                        "Data (MB)": st.column_config.NumberColumn("Data (MB)", format="%d"),
                        "Data (GB)": st.column_config.NumberColumn("Data (GB)", format="%.2f"),
                        "Price ($)": st.column_config.NumberColumn("Price ($)", format="$%.2f"),
                        "Voice": st.column_config.NumberColumn("Voice Minutes", format="%d")
                    }
                )
                
                # Add summary statistics for this category
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Offers", len(display_df))
                with col2:
                    avg_price = display_df['Price ($)'].mean()
                    st.metric("Avg Price", f"${avg_price:.2f}")
                with col3:
                    avg_data = display_df['Data (MB)'].mean()
                    st.metric("Avg Data", f"{avg_data:,.0f} MB")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #5F6368; padding: 20px;'>
            <small>Touch Recommendation Dashboard v1.0 | Data as of 2024 | Built with Streamlit</small>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()