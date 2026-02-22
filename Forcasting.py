
import streamlit as st
import requests
import pandas as pd
import plotly.express as px

# --- Dashboard Metadata ---
st.set_page_config(page_title="Gaza Solar SEMS", page_icon="🌤️", layout="wide")

# --- FORCED LIGHT THEME & CONTRAST CSS ---
st.markdown("""
    <style>
    /* Force Light Background for the whole app */
    .stApp {
        background-color: #F8F9FB !important;
    }

    /* Global Text Color: FORCED DARK GREY/BLACK */
    h1, h2, h3, h4, p, span, label, .stMarkdown {
        color: #1A1A1B !important;
    }

    /* --- ORANGE BUTTON STYLING (The requested change) --- */
    div.stButton > button {
        background-color: #FF9800 !important;
        color: white !important;
        border: none !important;
        font-weight: bold !important;
        padding: 0.5rem 1rem !important;
        transition: 0.3s !important;
    }

    div.stButton > button:hover {
        background-color: #E68900 !important;
        color: white !important;
        border: none !important;
    }

    /* Metric Cards: White Background + Shadow + DARK TEXT */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E0E0E0 !important;
        padding: 20px !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05) !important;
    }

    /* Forced Dark Color for Metric Labels and Values */
    div[data-testid="stMetricLabel"] > div {
        color: #4A4A4A !important; /* Grey for label */
        font-weight: 600 !important;
    }
    div[data-testid="stMetricValue"] > div {
        color: #004AAD !important; /* Professional Blue for values */
        font-size: 2rem !important;
    }

    /* Weather Status Box with distinct background and border */
    .status-box {
        background-color: #FFFFFF !important;
        padding: 25px !important;
        border-radius: 15px !important;
        border-left: 8px solid #FF9800 !important;
        margin: 20px 0 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
    }

    .status-box h3 {
        color: #FF9800 !important;
        margin-bottom: 10px !important;
    }

    .status-box p {
        color: #333333 !important;
        font-size: 1.1rem !important;
    }

    /* Sidebar Fix */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #EEE !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Header Section ---
st.title("☀️ Gaza Strip Solar Energy Predictor")
st.write("Professional Satellite-based Forecast for Solar PV Systems.")

# --- Geographic Data ---
districts = {
    "Gaza City": {"lat": 31.50, "lon": 34.46},
    "Deir al-Balah": {"lat": 31.41, "lon": 34.34},
    "Khan Yunis": {"lat": 31.34, "lon": 34.30},
    "Rafah": {"lat": 31.28, "lon": 34.25}
}

# --- Sidebar Inputs ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3222/3222800.png", width=80)
    st.header("Settings")
    selected_district = st.selectbox("Select District", list(districts.keys()))
    horizon = st.slider("Forecast Range (Days)", 1, 7, 3)
    st.divider()
    # The Button now follows the orange CSS styling
    process_btn = st.button("🚀 Analyze Now", use_container_width=True)

# --- Classification Logic ---
def classify_state(clouds, rain, rad):
    if rain > 50: return "🌧️ Rainy / Stormy", "High rainfall expected. Minimal solar harvest possible."
    elif clouds > 80: return "☁️ Heavy Overcast", "Low direct sunlight. Expect lower PV system efficiency."
    elif rad > 480 and clouds < 25: return "☀️ Optimal Conditions", "Clear sky. Maximum irradiance for peak energy harvest."
    else: return "🌤️ Partly Cloudy", "Stable energy window with minor cloud fluctuations."

if process_btn:
    coord = districts[selected_district]
    api_url = f"https://api.open-meteo.com/v1/forecast?latitude={coord['lat']}&longitude={coord['lon']}&hourly=temperature_2m,relative_humidity_2m,shortwave_radiation,precipitation_probability,cloud_cover&forecast_days={horizon}"

    try:
        response = requests.get(api_url).json()
        idx = 12 if horizon == 1 else 36

        t = response['hourly']['temperature_2m'][idx]
        h = response['hourly']['relative_humidity_2m'][idx]
        r = response['hourly']['shortwave_radiation'][idx]
        rn = response['hourly']['precipitation_probability'][idx]
        c = response['hourly']['cloud_cover'][idx]
        ts = response['hourly']['time'][idx]

        # Predictive Model Logic
        energy_yield = (t * 0.1) - (h * 0.02) + (r * 0.005)
        energy_yield = max(0, energy_yield)

        status_title, status_msg = classify_state(c, rn, r)

        # --- Dashboard Metrics ---
        st.subheader(f"📍 Results for {selected_district}")
        st.caption(f"Valid for: {ts.replace('T', ' ')}")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Predicted Yield", f"{energy_yield:.2f} kWh/m²")
        col2.metric("Irradiance", f"{r} W/m²")
        col3.metric("Temperature", f"{t} °C")
        col4.metric("Humidity", f"{h}%")

        # --- Status Highlight Box ---
        st.markdown(f"""
        <div class="status-box">
            <h3>{status_title}</h3>
            <p>{status_msg}</p>
        </div>
        """, unsafe_allow_html=True)

        # --- Charts ---
        st.subheader("📊 24-Hour Irradiance Trend")
        plot_df = pd.DataFrame({
            "Time": response['hourly']['time'][idx:idx+24],
            "Radiation (W/m²)": response['hourly']['shortwave_radiation'][idx:idx+24]
        }).set_index("Time")

        st.area_chart(plot_df, color="#FFA000")

    except Exception as err:
        st.error(f"Error: {err}")

st.markdown("---")
st.caption("Gaza Solar SEMS Dashboard | English Version 2.4 | Fixed Orange Action Button")
