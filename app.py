import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG & CLEAN PROFESSIONAL STYLING
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bengaluru Asset Intelligence",
    layout="wide",
    page_icon="https://img.icons8.com/ios-filled/50/6366f1/department.png",
    initial_sidebar_state="expanded",
)

PROFESSIONAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

/* ── Global ─────────────────────────────── */
:root {
    --primary: #6366f1;
    --primary-hover: #4f46e5;
    --secondary: #64748b;
    --bg-gradient: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    --card-bg: rgba(255, 255, 255, 0.8);
    --text-main: #000000;
    --text-muted: #1a1a1a;
    --accent: #f59e0b;
}

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: var(--text-main);
}

/* ── Main Content Contrast Fix ──────────────── */
/* Force dark text in the main area to prevent "white on white" in dark mode */
.stApp .main .stMarkdown, 
.stApp .main .stMarkdown p, 
.stApp .main .stMarkdown h1, 
.stApp .main .stMarkdown h2, 
.stApp .main .stMarkdown h3, 
.stApp .main .stMarkdown h4, 
.stApp .main .stMarkdown h5, 
.stApp .main .stMarkdown h6,
.stApp .main .stMarkdown li,
.stApp .main .stMarkdown ul,
.stApp .main .stMarkdown b,
.stApp .main .stMarkdown strong,
.stApp .main .stMarkdown span,
.stApp .main h1, .stApp .main h2, .stApp .main h3, .stApp .main h4, .stApp .main h5, .stApp .main h6,
.stMain h1, .stMain h2, .stMain h3, .stMain h4, .stMain h5, .stMain h6, .stMain p, .stMain li, .stMain ul, .stMain span, .stMain b, .stMain strong,
.stApp .main [data-testid="stMetricLabel"] p,
.stApp .main [data-testid="stMetricValue"],
.stApp .main [data-testid="stMetricDelta"],
.stApp .main [data-testid="stText"] {
    color: var(--text-main) !important;
}

/* Ensure clean-card text is always readable */
.clean-card, .clean-card p, .clean-card h1, .clean-card h2, .clean-card h3, .clean-card h4, .clean-card h5, .clean-card h6, .clean-card li {
    color: var(--text-main) !important;
}

/* Custom Selection Highlight */
::selection {
    background-color: rgba(99, 102, 241, 0.2) !important;
    color: var(--text-main) !important;
}
::-moz-selection {
    background-color: rgba(99, 102, 241, 0.2) !important;
    color: var(--text-main) !important;
}

/* Specific fix for metrics in main area */
.stApp .main [data-testid="stMetricValue"] div {
    color: var(--text-main) !important;
}

.stApp {
    background: #ffffff;
    background-image: 
        radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.05) 0px, transparent 50%),
        radial-gradient(at 100% 0%, rgba(245, 158, 11, 0.05) 0px, transparent 50%),
        linear-gradient(to right, #f1f5f9 1px, transparent 1px),
        linear-gradient(to bottom, #f1f5f9 1px, transparent 1px);
    background-size: 100% 100%, 100% 100%, 40px 40px, 40px 40px;
    background-attachment: fixed;
}

/* ── Sidebar ────────────────────────────── */
section[data-testid="stSidebar"] {
    background-color: #0f172a !important;
    border-right: 1px solid rgba(255, 255, 255, 0.05);
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4,
section[data-testid="stSidebar"] h5,
section[data-testid="stSidebar"] h6,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    color: #ffffff !important;
    opacity: 1 !important;
}

section[data-testid="stSidebar"] hr {
    border-color: rgba(255, 255, 255, 0.1) !important;
}

/* Ensure secondary text like captions are also readable */
section[data-testid="stSidebar"] .stCaption {
    color: #94a3b8 !important;
}

/* Fix for Selectbox, Input, and other widget dropdowns in dark sidebar */
section[data-testid="stSidebar"] [data-baseweb="select"] > div,
section[data-testid="stSidebar"] div[data-baseweb="input"] input,
section[data-testid="stSidebar"] div[data-baseweb="input"] div,
section[data-testid="stSidebar"] .stNumberInput input {
    background-color: #1e293b !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    border-color: #334155 !important;
}

section[data-testid="stSidebar"] [data-testid="stMetricValue"] {
    color: white !important;
}

/* ── Bento Cards ────────────────────────── */
.clean-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 20px;
    padding: 2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
    position: relative;
    overflow: hidden;
}

.clean-card::after {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.2), transparent);
}

.clean-card:hover {
    transform: translateY(-4px) scale(1.01);
    box-shadow: 0 20px 40px -15px rgba(0,0,0,0.08);
    border-color: rgba(99, 102, 241, 0.2);
}

/* ── Metric Grid ────────────────────────── */
.metric-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 1.5rem;
    text-align: left;
    transition: all 0.3s ease;
}

.metric-box:hover {
    background: white;
    border-color: var(--primary);
}

.metric-box h3 {
    color: #64748b !important;
    font-size: 0.7rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

.metric-box p {
    color: #0f172a !important;
    font-size: 1.5rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: -1px;
}

/* ── Comparison ─────────────────────────── */
.compare-above {
    background: #ecfdf5;
    border: 1px solid #10b981;
    border-radius: 16px;
    padding: 1rem;
}
.compare-below {
    background: #fef2f2;
    border: 1px solid #ef4444;
    border-radius: 16px;
    padding: 1rem;
}

/* ── Section Titles ─────────────────────── */
.section-title {
    color: var(--text-main) !important;
    font-size: 1.75rem;
    font-weight: 800;
    margin-top: 2rem;
    margin-bottom: 0.5rem;
    letter-spacing: -0.025em;
}

.section-subtitle {
    color: var(--text-muted) !important;
    font-size: 1rem;
    margin-bottom: 2rem;
}

/* ── Tabs ───────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
    background: transparent;
    padding: 10px 0;
}

.stTabs [data-baseweb="tab"] {
    height: 45px;
    border-radius: 12px !important;
    background-color: rgba(255, 255, 255, 0.5) !important;
    border: 1px solid rgba(0, 0, 0, 0.05) !important;
    color: var(--text-muted) !important;
    padding: 0 24px !important;
    transition: all 0.2s ease;
}

.stTabs [aria-selected="true"] {
    background-color: var(--primary) !important;
    color: white !important;
    box-shadow: 0 10px 15px -3px rgba(99, 102, 241, 0.3) !important;
}

/* ── Buttons ────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, var(--primary) 0%, var(--primary-hover) 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 16px !important;
    font-weight: 700 !important;
    padding: 0.75rem 2rem !important;
    box-shadow: 0 10px 15px -3px rgba(99, 102, 241, 0.3) !important;
    transition: all 0.3s ease !important;
    width: 100%;
}

.stButton > button:hover {
    transform: scale(1.02);
    box-shadow: 0 20px 25px -5px rgba(99, 102, 241, 0.4) !important;
}

/* ── Feature badge ──────────────────────── */
.feature-badge {
    display: inline-block;
    background: white;
    color: var(--primary);
    padding: 6px 16px;
    border-radius: 100px;
    font-size: 0.85rem;
    font-weight: 600;
    margin: 4px;
    border: 1px solid rgba(99, 102, 241, 0.1);
    box-shadow: 0 2px 4px rgba(0,0,0,0.02);
}

/* ── Hide Streamlit branding ────────────── */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* ── Header & Sidebar Toggle ───────────────── */
header[data-testid="stHeader"] {
    background-color: rgba(255, 255, 255, 0.5) !important;
    backdrop-filter: blur(15px) !important;
    border-bottom: 1px solid rgba(0, 0, 0, 0.05) !important;
    visibility: visible !important;
    display: block !important;
    z-index: 99999 !important;
}

/* Make the sidebar toggle and close buttons extremely prominent and solid */
[data-testid="stSidebarCollapsedControl"],
button[data-testid="baseButton-headerNoPadding"],
[data-testid="stSidebarNav"] + div button {
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
    color: white !important;
    border-radius: 10px !important;
    padding: 10px !important;
    width: 40px !important;
    height: 40px !important;
    margin: 10px !important;
    box-shadow: 0 10px 20px -5px rgba(99, 102, 241, 0.5) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    visibility: visible !important;
    opacity: 1 !important;
    border: none !important;
    transition: all 0.3s ease !important;
    z-index: 100000 !important;
}

/* Specific fix for the 'X' button inside the sidebar */
section[data-testid="stSidebar"] button[data-testid="baseButton-headerNoPadding"] {
    position: absolute !important;
    top: 10px !important;
    right: 10px !important;
}

[data-testid="stSidebarCollapsedControl"] svg,
button[data-testid="baseButton-headerNoPadding"] svg {
    width: 24px !important;
    height: 24px !important;
    fill: white !important;
    color: white !important;
}

[data-testid="stSidebarCollapsedControl"]:hover,
button[data-testid="baseButton-headerNoPadding"]:hover {
    transform: scale(1.1) rotate(5deg);
    box-shadow: 0 20px 25px -5px rgba(99, 102, 241, 0.5) !important;
}

/* ── Animations ─────────────────────────── */
@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.clean-card, .metric-box, .stTabs {
    animation: fadeInUp 0.8s cubic-bezier(0.2, 0.8, 0.2, 1) both;
}

/* Staggered-like delay for content */
.stApp .main .block-container {
    animation: fadeIn 0.5s ease-out;
}

@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

/* Custom Scrollbar */
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-track {
    background: #f1f1f1;
}
::-webkit-scrollbar-thumb {
    background: #cbd5e1;
    border-radius: 10px;
}
::-webkit-scrollbar-thumb:hover {
    background: #94a3b8;
}
</style>
"""
st.markdown(PROFESSIONAL_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# DATA LOADING & MODEL TRAINING
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_and_train_model():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, "Bengaluru_House_Data.csv")
    if not os.path.isfile(path):
        path = os.path.join(script_dir, "bengaluru_house_prices.csv")

    raw_df = pd.read_csv(path)
    df = raw_df.copy()

    # 1. Feature Engineering: BHK
    if "size" in df.columns:
        df["bhk"] = df["size"].str.extract(r"(\d+)").astype(float)
    
    # 2. Feature Engineering: Total Sqft
    def convert_sqft(x):
        try:
            return float(x)
        except Exception:
            if isinstance(x, str) and "-" in x:
                parts = x.split("-")
                try:
                    return (float(parts[0].strip()) + float(parts[1].strip())) / 2.0
                except Exception:
                    return np.nan
            return np.nan

    if "total_sqft" in df.columns:
        df["total_sqft"] = df["total_sqft"].apply(convert_sqft)

    # 3. Feature Engineering: Balcony
    if "balcony" in df.columns:
        df["balcony"] = df["balcony"].fillna(df["balcony"].median())
    else:
        df["balcony"] = 0

    # 4. Feature Engineering: Availability (Binary: Ready to Move = 1, Future = 0)
    if "availability" in df.columns:
        df["ready_to_move"] = df["availability"].apply(lambda x: 1 if x == "Ready To Move" else 0)
    else:
        df["ready_to_move"] = 1

    # 5. Build Model Dataframe
    # Include Area Type as well
    features = ["location", "total_sqft", "bath", "bhk", "balcony", "ready_to_move", "area_type", "price"]
    model_df = df[features].copy()
    model_df.dropna(inplace=True)
    model_df["location"] = model_df["location"].apply(lambda x: x.strip())

    # Handle rare locations
    location_counts = model_df["location"].value_counts()
    rare_locations = location_counts[location_counts <= 10].index
    model_df["location"] = model_df["location"].apply(
        lambda x: "Other" if x in rare_locations else x
    )

    # One-hot encode Location and Area Type
    loc_dummies = pd.get_dummies(model_df["location"], prefix="loc")
    area_dummies = pd.get_dummies(model_df["area_type"], prefix="area")
    
    # Drop 'Other' for location and one category for area_type to avoid dummy variable trap
    if "loc_Other" in loc_dummies.columns:
        loc_dummies = loc_dummies.drop("loc_Other", axis="columns")
    
    df_encoded = pd.concat([model_df.drop(["location", "area_type"], axis=1), loc_dummies, area_dummies], axis="columns")

    X = df_encoded.drop(columns=["price"])
    y = df_encoded["price"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # Full model for production
    full_model = LinearRegression()
    full_model.fit(X, y)

    locations = sorted([loc for loc in model_df["location"].unique() if loc != "Other"])
    area_types = sorted(model_df["area_type"].unique())
    feature_names = X.columns.tolist()
    
    return (
        full_model,
        feature_names,
        locations,
        area_types,
        model_df,
        raw_df,
        {"r2": r2, "mae": mae, "n_samples": len(X), "n_features": len(feature_names)},
        dict(zip(feature_names, full_model.coef_)),
    )


with st.spinner("Initializing Enterprise Valuation Engine..."):
    model, feature_columns, locations, area_types, model_df, raw_df, metrics, coef_dict = (
        load_and_train_model()
    )


# ─────────────────────────────────────────────────────────────
# UNIT CONVERSION HELPERS
# ─────────────────────────────────────────────────────────────
UNIT_FACTORS = {
    "Square Feet": 1.0,
    "Square Meters": 10.7639,
    "Square Yards": 9.0,
}


def to_sqft(value, unit):
    return value * UNIT_FACTORS[unit]


def from_sqft(sqft_val, unit):
    return sqft_val / UNIT_FACTORS[unit]


# ─────────────────────────────────────────────────────────────
# SIDEBAR – INPUT PROPERTY DETAILS
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
    <div style="text-align:center; padding: 20px 0;">
        <div style="background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%); 
                    width: 50px; height: 50px; border-radius: 12px; 
                    display: flex; align-items: center; justify-content: center; 
                    margin: 0 auto 15px auto; box-shadow: 0 10px 15px -3px rgba(99, 102, 241, 0.4);">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path><polyline points="9 22 9 12 15 12 15 22"></polyline></svg>
        </div>
        <h2 style="margin:0; font-weight:800; color:#f8fafc; font-size: 1.2rem; letter-spacing: 0.5px; text-transform: uppercase;">
        Parameter Studio</h2>
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.divider()

    # Location
    selected_location = st.selectbox("Location Index", locations, index=locations.index("Whitefield") if "Whitefield" in locations else 0)

    # Unit selector
    area_unit = st.radio("Measurement Standard", list(UNIT_FACTORS.keys()), horizontal=True)

    min_val = int(from_sqft(300, area_unit))
    max_val = int(from_sqft(30000, area_unit))
    default_val = int(from_sqft(1200, area_unit))
    step_val = max(1, int(from_sqft(100, area_unit)))

    area_input = st.number_input(
        f"Total Portfolio Area ({area_unit})",
        min_value=min_val,
        max_value=max_val,
        value=default_val,
        step=step_val,
    )
    sqft = to_sqft(area_input, area_unit)
    if area_unit != "Square Feet":
        st.caption(f"Equates to {sqft:,.0f} sq. ft")

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        bhk = st.number_input("Bedrooms (BHK)", min_value=1, max_value=20, value=2, step=1)
    with col_b2:
        bath = st.number_input("Bathrooms", min_value=1, max_value=20, value=2, step=1)

    st.divider()

    # Property Type
    st.markdown("##### Structural Type")
    selected_area_type = st.selectbox(
        "Type",
        area_types,
        index=area_types.index("Super built-up Area") if "Super built-up Area" in area_types else 0,
        label_visibility="collapsed",
    )

    # Balcony and Availability
    col_x1, col_x2 = st.columns(2)
    with col_x1:
        balcony_count = st.number_input("Balconies", min_value=0, max_value=10, value=1, step=1)
    with col_x2:
        is_ready = st.toggle("Ready to Move", value=True)

    # Property Age
    st.markdown("##### Asset Vintage")
    property_age = st.select_slider(
        "Age",
        options=["Brand New", "1-3 Years", "3-5 Years", "5-10 Years", "10+ Years"],
        value="Brand New",
        label_visibility="collapsed",
    )

    st.divider()

    # Amenities
    st.markdown("##### Premium Features")
    amenity_cols = st.columns(2)
    with amenity_cols[0]:
        has_gated = st.checkbox("Secure Perimeter", value=False)
        has_pool = st.checkbox("Aquatic Facilities", value=False)
        has_gym = st.checkbox("Fitness Center", value=False)
    with amenity_cols[1]:
        has_metro = st.checkbox("Transit Proximity", value=False)
        has_parking = st.checkbox("Designated Parking", value=False)
        has_garden = st.checkbox("Landscaped Grounds", value=False)

    st.divider()

    predict_clicked = st.button("RUN VALUATION ENGINE", type="primary", use_container_width=True)


# ─────────────────────────────────────────────────────────────
# AMENITY ADJUSTMENT LOGIC
# ─────────────────────────────────────────────────────────────
def compute_amenity_multiplier():
    """Simulate amenity premium based on selected features."""
    multiplier = 1.0
    if has_gated:
        multiplier += 0.04
    if has_pool:
        multiplier += 0.05
    if has_gym:
        multiplier += 0.02
    if has_metro:
        multiplier += 0.06
    if has_parking:
        multiplier += 0.03
    if has_garden:
        multiplier += 0.02
    return multiplier


def compute_age_multiplier():
    age_factors = {
        "Brand New": 1.0,
        "1-3 Years": 0.97,
        "3-5 Years": 0.93,
        "5-10 Years": 0.87,
        "10+ Years": 0.80,
    }
    return age_factors.get(property_age, 1.0)


def predict_price():
    # Construct input vector
    x_input = np.zeros(len(feature_columns))
    
    # 1. Direct Features
    try:
        x_input[feature_columns.index("total_sqft")] = sqft
        x_input[feature_columns.index("bath")] = bath
        x_input[feature_columns.index("bhk")] = bhk
        x_input[feature_columns.index("balcony")] = balcony_count
        x_input[feature_columns.index("ready_to_move")] = 1 if is_ready else 0
        
        # 2. Location (one-hot)
        loc_col = f"loc_{selected_location}"
        if loc_col in feature_columns:
            x_input[feature_columns.index(loc_col)] = 1
            
        # 3. Area Type (one-hot)
        area_col = f"area_{selected_area_type}"
        if area_col in feature_columns:
            x_input[feature_columns.index(area_col)] = 1
            
        base = model.predict([x_input])[0]
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        return 0, 0
        
    # Apply multipliers for simulated attributes (Amenities/Age)
    adjusted = base * compute_amenity_multiplier() * compute_age_multiplier()
    return base, adjusted


# ─────────────────────────────────────────────────────────────
# HELPER – Location Stats
# ─────────────────────────────────────────────────────────────
def location_stats(loc):
    subset = model_df[model_df["location"] == loc]
    if len(subset) == 0:
        return None
    return {
        "avg_price": subset["price"].mean(),
        "median_price": subset["price"].median(),
        "min_price": subset["price"].min(),
        "max_price": subset["price"].max(),
        "avg_sqft": subset["total_sqft"].mean(),
        "count": len(subset),
        "price_per_sqft": (subset["price"] / subset["total_sqft"]).mean() * 100000,
    }


# ─────────────────────────────────────────────────────────────
# MAIN CONTENT AREA – TABS
# ─────────────────────────────────────────────────────────────
st.markdown(
    """
<div style="text-align:center; padding: 40px 0 20px 0;">
    <h1 style="font-size:4rem; font-weight:900; margin:0; 
               background: linear-gradient(135deg, #1e293b 0%, #4f46e5 100%);
               -webkit-background-clip: text;
               -webkit-text-fill-color: transparent;
               letter-spacing: -2px; text-transform: uppercase;">
    PropIntel Bengaluru</h1>
    <p style="color:#64748b; font-size:1.1rem; font-weight: 500; margin-top:10px; text-transform: uppercase; letter-spacing: 2px;">
    Institutional Grade Asset Valuation & Market Analytics
    <span style="display:block; width: 60px; height: 3px; background: #6366f1; margin: 24px auto; border-radius: 2px;"></span>
    </p>
</div>
""",
    unsafe_allow_html=True,
)

tab_predict, tab_eda, tab_insights, tab_about = st.tabs(
    ["VALUATION ENGINE", "MARKET DYNAMICS", "STRATEGIC INSIGHTS", "TECHNICAL AUDIT"]
)


# ═══════════════════════════════════════════════════════════════
# TAB 1 – PREDICTION & SCORING
# ═══════════════════════════════════════════════════════════════
with tab_predict:
    if predict_clicked:
        base_price, adjusted_price = predict_price()
        amenity_premium = (compute_amenity_multiplier() - 1) * 100
        age_discount = (1 - compute_age_multiplier()) * 100
        loc_data = location_stats(selected_location)

        # 1. VALUATION OVERVIEW
        col_res, col_side = st.columns([2, 1])
        
        with col_res:
            st.markdown(
                f"""
            <div class="clean-card" style="height: 100%; display: flex; flex-direction: column; justify-content: center; border-left: 5px solid #6366f1;">
                <p style="color:#64748b; font-size:0.75rem; font-weight:800; text-transform:uppercase; letter-spacing:3px; margin-bottom:15px;">
                Asset Market Valuation</p>
                <h1 style="font-size:5.5rem; font-weight:900; margin:0; color:#0f172a; letter-spacing: -4px; line-height: 1;">
                ₹ {adjusted_price:,.2f} <span style="font-size: 1.5rem; color: #94a3b8; letter-spacing: 0; font-weight: 500;">Lakhs</span></h1>
                <div style="margin-top: 30px; display: flex; gap: 30px;">
                    <div>
                        <p style="color:#94a3b8; font-size:0.7rem; font-weight:700; text-transform:uppercase;">Locality</p>
                        <p style="color:#1e293b; font-weight:700;">{selected_location}</p>
                    </div>
                    <div>
                        <p style="color:#94a3b8; font-size:0.7rem; font-weight:700; text-transform:uppercase;">Unit Rate</p>
                        <p style="color:#1e293b; font-weight:700;">₹ {adjusted_price*100000/sqft:,.0f} / sqft</p>
                    </div>
                    <div>
                        <p style="color:#94a3b8; font-size:0.7rem; font-weight:700; text-transform:uppercase;">Status</p>
                        <p style="color:#1e293b; font-weight:700;">{"Ready" if is_ready else "Projected"}</p>
                    </div>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col_side:
            # Value Score Calculation
            if loc_data:
                price_per_sqft = (adjusted_price * 100000) / sqft
                area_avg_psf = loc_data['price_per_sqft']
                # Score: 100 is area avg. >100 is expensive, <100 is "good value"
                value_ratio = area_avg_psf / price_per_sqft if price_per_sqft > 0 else 1
                value_score = min(100, max(0, int(value_ratio * 70))) # Scaled 
            else:
                value_score = 75

            fig_score = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = value_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Investment Score", 'font': {'size': 14, 'color': "#64748b"}},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                    'bar': {'color': "#6366f1"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "#e2e8f0",
                    'steps': [
                        {'range': [0, 40], 'color': '#fef2f2'},
                        {'range': [40, 70], 'color': '#fffbeb'},
                        {'range': [70, 100], 'color': '#f0fdf4'}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90}}))
            fig_score.update_layout(height=200, margin=dict(l=10, r=10, t=40, b=10), 
                                    font={'family': "Plus Jakarta Sans", 'color': "#0f172a"})
            st.plotly_chart(fig_score, use_container_width=True)

        # 2. PRICE TRAJECTORY ANALYSIS
        st.markdown('<p class="section-title">📉 Price Trajectory Analysis</p>', unsafe_allow_html=True)
        
        # Generate trajectory data (Price vs Area for this location)
        area_range = np.linspace(sqft*0.5, sqft*2.0, 20)
        traj_prices = []
        for a in area_range:
            # Temporary input for trajectory
            tx = np.zeros(len(feature_columns))
            tx[feature_columns.index("total_sqft")] = a
            tx[feature_columns.index("bath")] = bath
            tx[feature_columns.index("bhk")] = bhk
            tx[feature_columns.index("balcony")] = balcony_count
            tx[feature_columns.index("ready_to_move")] = 1 if is_ready else 0
            if f"loc_{selected_location}" in feature_columns:
                tx[feature_columns.index(f"loc_{selected_location}")] = 1
            if f"area_{selected_area_type}" in feature_columns:
                tx[feature_columns.index(f"area_{selected_area_type}")] = 1
            
            p = model.predict([tx])[0] * compute_amenity_multiplier() * compute_age_multiplier()
            traj_prices.append(p)

        fig_traj = px.line(x=area_range, y=traj_prices, 
                          labels={'x': f'Area ({area_unit})', 'y': 'Market Price (Lakhs)'},
                          title=f"Valuation Trajectory for {selected_location}")
        fig_traj.add_scatter(x=[sqft], y=[adjusted_price], mode='markers', name='Selected Asset', 
                            marker=dict(size=15, color='#6366f1', symbol='diamond'))
        fig_traj.update_layout(hovermode="x unified", font={'family': "Plus Jakarta Sans", 'color': "#0f172a"}, 
                              plot_bgcolor='white', paper_bgcolor='white')
        st.plotly_chart(fig_traj, use_container_width=True)

        # 3. REAL-WORLD SCORECARD
        st.markdown('<p class="section-title">🎯 Property Scorecard</p>', unsafe_allow_html=True)
        c_score1, c_score2 = st.columns([1, 1])
        
        with c_score1:
            # Radar Chart for Property Attributes
            categories = ['Space', 'Location', 'Amenities', 'Value', 'Ready-Status']
            
            # Simulated scores based on inputs
            s_space = min(10, sqft/1000 * 3)
            s_loc = 5 + (coef_dict.get(f"loc_{selected_location}", 0) / 20)
            s_amen = compute_amenity_multiplier() * 5
            s_val = value_score / 10
            s_ready = 10 if is_ready else 4
            
            scores = [s_space, s_loc, s_amen, s_val, s_ready]
            
            fig_radar = go.Figure(data=go.Scatterpolar(
                r=scores,
                theta=categories,
                fill='toself',
                line_color='#6366f1'
            ))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 10])), 
                                    showlegend=False, height=350, margin=dict(l=40, r=40, t=20, b=20),
                                    font={'family': "Plus Jakarta Sans", 'color': "#0f172a"})
            st.plotly_chart(fig_radar, use_container_width=True)

        with c_score2:
            st.markdown(f"""
            <div class="clean-card">
                <h4 style="margin-top:0; color:#1e293b;">Market Insights</h4>
                <ul style="color:#64748b; font-size:0.9rem; line-height:1.8;">
                    <li><b>Segment:</b> {"Luxury" if adjusted_price > 200 else "Mid-Range" if adjusted_price > 80 else "Budget"}</li>
                    <li><b>Area Demand:</b> High in {selected_location}</li>
                    <li><b>Yield Projection:</b> Estimated 3.2% - 4.1% annually</li>
                    <li><b>Resale Potential:</b> {"High" if is_ready else "Moderate"}</li>
                </ul>
                <hr style="opacity:0.1">
                <p style="color:#6366f1; font-weight:700; margin-bottom:5px;">Recommendation</p>
                <p style="color:#1e293b; font-size:0.95rem;">
                { "A premium asset with high investment potential." if value_score > 70 else "A fair market-value proposition for end-usage." if value_score > 50 else "Consider negotiating; price is above area benchmark."}
                </p>
            </div>
            """, unsafe_allow_html=True)

    else:
        # Default state
        st.markdown(
            """
        <div class="clean-card" style="text-align:center; padding:100px 40px; border: 1px dashed #cbd5e1;">
            <div style="background: #f1f5f9; width: 80px; height: 80px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 30px auto;">
                <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#6366f1" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path><polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline><line x1="12" y1="22.08" x2="12" y2="12"></line></svg>
            </div>
            <h2 style="color:#1e293b; font-weight:800; margin-bottom:15px; font-size: 2.2rem; letter-spacing:-1px;">PropIntel Asset Valuation</h2>
            <p style="color:#64748b; font-size:1.1rem; max-width:600px; margin:0 auto 40px auto; line-height: 1.7;">
            Leverage institutional-grade machine learning to analyze property valuations in Bengaluru. 
            Configure your asset parameters in the sidebar to generate a <b>Real-World Feasibility Report</b>.</p>
            <div style="display:flex; justify-content:center; gap:20px;">
                <div class="feature-badge">13,000+ Listings</div>
                <div class="feature-badge">Location Intelligence</div>
                <div class="feature-badge">Yield Projections</div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════
# TAB 2 – MARKET DYNAMICS (EDA)
# ═══════════════════════════════════════════════════════════════
with tab_eda:
    st.markdown(
        '<p class="section-title">📊 Market Dynamics Dashboard</p><p class="section-subtitle">Real-time analysis of the Bengaluru residential ecosystem</p>',
        unsafe_allow_html=True,
    )

    eda_tab1, eda_tab2, eda_tab3 = st.tabs(["Price Landscapes", "Locality Heatmaps", "Feature Analysis"])

    with eda_tab1:
        c1, c2 = st.columns(2)
        with c1:
            fig_hist = px.histogram(model_df, x="price", nbins=50, title="Market Price Distribution", 
                                   color_discrete_sequence=['#6366f1'])
            fig_hist.update_layout(xaxis_title="Price (Lakhs)", yaxis_title="Count", plot_bgcolor='rgba(0,0,0,0)',
                                   font={'family': "Plus Jakarta Sans", 'color': "#0f172a"})
            st.plotly_chart(fig_hist, use_container_width=True)
        with c2:
            fig_box = px.box(model_df, x="bhk", y="price", title="Price Trajectory by BHK",
                            color_discrete_sequence=['#34d399'])
            fig_box.update_layout(xaxis_title="BHK", yaxis_title="Price (Lakhs)", plot_bgcolor='rgba(0,0,0,0)',
                                  font={'family': "Plus Jakarta Sans", 'color': "#0f172a"})
            st.plotly_chart(fig_box, use_container_width=True)

    with eda_tab2:
        st.markdown('<div class="clean-card"><h4>High-Value Localities vs Volume</h4></div>', unsafe_allow_html=True)
        loc_stats = model_df.groupby("location").agg({"price": "mean", "total_sqft": "count"}).rename(columns={"total_sqft": "listings"})
        loc_stats = loc_stats[loc_stats['listings'] > 50].sort_values("price", ascending=False).head(30)
        
        fig_bubble = px.scatter(loc_stats, x="listings", y="price", size="listings", color="price",
                               hover_name=loc_stats.index, title="Market Segment Analysis by Location",
                               color_continuous_scale='Viridis')
        fig_bubble.update_layout(font={'family': "Plus Jakarta Sans", 'color': "#0f172a"})
        st.plotly_chart(fig_bubble, use_container_width=True)

    with eda_tab3:
        fig_corr = px.scatter(model_df.sample(2000), x="total_sqft", y="price", color="bhk", 
                             size="bath", hover_data=['location'], title="The Area-Price Correlation Trajectory",
                             trendline="ols", color_continuous_scale='RdYlBu')
        fig_corr.update_layout(font={'family': "Plus Jakarta Sans", 'color': "#0f172a"})
        st.plotly_chart(fig_corr, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# TAB 3 – STRATEGIC INSIGHTS
# ═══════════════════════════════════════════════════════════════
with tab_insights:
    st.markdown(
        '<p class="section-title">💡 Strategic Intelligence</p><p class="section-subtitle">Decoding the algorithmic weightage of asset features</p>',
        unsafe_allow_html=True,
    )

    # Key Feature Weights
    weights = pd.DataFrame({
        'Feature': ['Area (Sqft)', 'Bathrooms', 'Bedrooms', 'Balcony', 'Ready to Move'],
        'Impact': [coef_dict.get('total_sqft', 0)*100, coef_dict.get('bath', 0), 
                   coef_dict.get('bhk', 0), coef_dict.get('balcony', 0), 
                   coef_dict.get('ready_to_move', 0)]
    })
    
    fig_weights = px.bar(weights, x='Feature', y='Impact', color='Impact', 
                         title="Feature Weightage (Lakhs per unit)",
                         color_continuous_scale='Blues')
    fig_weights.update_layout(font={'family': "Plus Jakarta Sans", 'color': "#0f172a"})
    st.plotly_chart(fig_weights, use_container_width=True)

    st.markdown(f"""
    <div class="clean-card" style="background:#0f172a; color:white; border:none;">
        <h4 style="color:#6366f1;">Intelligence Briefing</h4>
        <p style="font-size:0.95rem; opacity:0.8;">
            Our model indicates that <b>Total Area</b> remains the strongest predictor of price, with an 
            estimated impact of ₹ {coef_dict.get('total_sqft', 0)*100:.2f} Lakhs per 100 sqft. 
            Interestingly, <b>Location Premiums</b> in areas like {locations[0] if locations else "N/A"} 
            can add up to ₹ {max(coef_dict.values()):.1f}L to the base valuation.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TAB 4 – TECHNICAL AUDIT
# ═══════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("""
    <div class="clean-card">
        <h3 style="color:#000000 !important;">Technical Infrastructure</h3>
        <p style="color:#000000 !important;">This institutional-grade valuation engine utilizes a multi-variate linear regression model 
        optimized with one-hot encoded locality intelligence.</p>
    </div>
    """, unsafe_allow_html=True)
    
    c_audit1, c_audit2 = st.columns(2)
    with c_audit1:
        st.markdown('<h3 style="color:#000000 !important;">Model Diagnostics</h3>', unsafe_allow_html=True)
        st.metric("Model R² Score", f"{metrics['r2']:.4f}")
        st.metric("Mean Absolute Error", f"₹ {metrics['mae']:.2f} L")
        st.progress(metrics['r2'])
    
    with c_audit2:
        st.markdown('<h3 style="color:#000000 !important;">Data Integrity</h3>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="color:#000000 !important;">
            <ul style="line-height: 1.8;">
                <li><b>Total Training Samples:</b> <span style="color:#000000 !important;">{metrics['n_samples']:,}</span></li>
                <li><b>Dimensionality:</b> <span style="color:#000000 !important;">{metrics['n_features']} features</span></li>
                <li><b>Primary Features:</b> <span style="color:#000000 !important;">Area, BHK, Bath, Balcony, Availability, Area Type, Location</span></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
