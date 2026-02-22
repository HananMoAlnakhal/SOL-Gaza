import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import folium
from streamlit_folium import folium_static

# --- 1. Page Config & Theme ---
st.set_page_config(
    page_title="Gaza Solar Intelligence",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a polished look
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    h1, h2, h3 { color: #2c3e50; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. Data Loading & Processing ---
@st.cache_data
def load_and_process_data():
    # Load data
    url = 'https://docs.google.com/spreadsheets/d/13dWUGh5gmRUBLvEp8hsmgbcbrUYqZu_Ln2CyA7ViRUw/export?format=csv'
    df = pd.read_csv(url)

    # Cleaning
    df.replace(-999.00, np.nan, inplace=True)
    df.dropna(subset=['ALLSKY_SFC_SW_DWN'], inplace=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    df['Date'] = pd.to_datetime(df['Date'])

    # Engineering
    le = LabelEncoder()
    df['City_ID'] = le.fit_transform(df['City'])

    features = ['ALLSKY_SFC_SW_DWN', 'CLRSKY_SFC_SW_DWN', 'WS2M', 'WS10M', 'RH2M', 'PRECTOTCORR', 'T2M', 'City_ID']
    X = df[features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    # PCA
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)
    for i in range(3):
        df[f'PCA{i+1}'] = X_pca[:, i]

    # Dynamic Naming
    cluster_order = df.groupby('Cluster')['ALLSKY_SFC_SW_DWN'].mean().sort_values(ascending=False).index
    mapping = {cluster_id: f"Level {rank+1}" for rank, cluster_id in enumerate(cluster_order)}
    df['Recommendation'] = df['Cluster'].map(mapping)

    return df, X_scaled, features

df, X_scaled, feature_list = load_and_process_data()

# --- 3. Sidebar Navigation ---
with st.sidebar:
    st.title("☀️ Solar Intel")
    st.image("https://img.icons8.com/fluency/96/000000/sun.png", width=80)
    page = st.radio("Navigate View", ["Strategic Overview", "Clustering Analysis", "Raw Data Explorer"])
    st.divider()
    st.info("Source: NASA POWER Project")

# --- 4. Page Logic ---

if page == "Strategic Overview":
    st.title("📊 Perception & Pattern Discovery")
    st.markdown("Analyzing solar irradiance trends across the Gaza Strip.")

    # KPI Row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Avg Irradiance", f"{df['ALLSKY_SFC_SW_DWN'].mean():.2f}", "kWh/m²/d")
    m2.metric("Max Potential", f"{df['ALLSKY_SFC_SW_DWN'].max():.2f}", "Peak")
    m3.metric("Cities Covered", df['City'].nunique())
    m4.metric("Data Points", len(df))

    tab1, tab2 = st.tabs(["📈 Temporal Trends", "🗺️ Geospatial Analysis"])

    with tab1:
        st.subheader("Temporal Analysis (Gaza City)")
        gaza_city = df[df['City'] == 'Gaza'].sort_values('Date')
        gaza_city['MA30'] = gaza_city['ALLSKY_SFC_SW_DWN'].rolling(window=30).mean()

        fig_temp = go.Figure()
        fig_temp.add_trace(go.Scatter(x=gaza_city['Date'], y=gaza_city['ALLSKY_SFC_SW_DWN'],
                                     name='Daily Actual', line=dict(color='orange', width=1), opacity=0.4))
        fig_temp.add_trace(go.Scatter(x=gaza_city['Date'], y=gaza_city['MA30'],
                                     name='30-Day Trend', line=dict(color='red', width=3)))
        fig_temp.update_layout(template="plotly_white", hovermode="x unified", height=500)
        st.plotly_chart(fig_temp, use_container_width=True)

    with tab2:
        col_left, col_right = st.columns([1, 1])
        with col_left:
            st.subheader("Irradiance Distribution")
            fig_box = px.box(df, x='City', y='ALLSKY_SFC_SW_DWN', color='City',
                             points="all", template="plotly_white", title="Regional Variance")
            st.plotly_chart(fig_box, use_container_width=True)

        with col_right:
            st.subheader("Geographical Markers")
            corrected_locations = {
                "Gaza City": [31.5000, 34.4667], "Deir al-Balah": [31.4171, 34.3531],
                "Khan Yunis": [31.3462, 34.3061], "Rafah": [31.2847, 34.2534]
            }
            m = folium.Map(location=[31.35, 34.40], zoom_start=10)
            for city, coords in corrected_locations.items():
                folium.Marker(coords, popup=city, icon=folium.Icon(color='orange', icon='sun', prefix='fa')).add_to(m)
            folium_static(m, width=500)

elif page == "Clustering Analysis":
    st.title("🤖 Machine Learning Insights")

    # Sidebar control for Silhouette
    st.sidebar.subheader("ML Controls")
    show_silhouette = st.sidebar.checkbox("Show Elbow Analysis", value=False)

    if show_silhouette:
        k_range = range(2, 11)
        scores = [silhouette_score(X_scaled, KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_scaled)) for k in k_range]
        fig_k = px.line(x=k_range, y=scores, markers=True, title="Optimal Cluster Selection (Silhouette Method)")
        st.plotly_chart(fig_k, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("2D PCA Projection")
        fig_pca = px.scatter(df, x='PCA1', y='PCA2', color='Recommendation',
                             hover_data=['Date', 'City'], template='plotly_white')
        st.plotly_chart(fig_pca, use_container_width=True)

    with c2:
        st.subheader("3D Cluster Cloud")
        fig_3d = px.scatter_3d(df, x='PCA1', y='PCA2', z='PCA3', color='Recommendation',
                               size_max=5, opacity=0.7, template='plotly_dark')
        st.plotly_chart(fig_3d, use_container_width=True)

    st.success(f"Silhouette Score for K=3: **{silhouette_score(X_scaled, df['Cluster']):.3f}**")

elif page == "Raw Data Explorer":
    st.title("📂 Data Explorer")
    selected_city = st.multiselect("Filter by City", options=df['City'].unique(), default=df['City'].unique())
    filtered_df = df[df['City'].isin(selected_city)]
    st.dataframe(filtered_df, use_container_width=True)
    st.download_button("Download CSV", filtered_df.to_csv().encode('utf-8'), "solar_data.csv", "text/csv")