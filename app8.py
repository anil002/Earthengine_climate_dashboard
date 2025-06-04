import ee
from matplotlib.pylab import f
import streamlit as st
from google.oauth2 import service_account
import json
import folium
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
from scipy.stats import linregress, skew, kurtosis
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import geopandas as gpd
import fiona
import zipfile
import tempfile
import os
import requests
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Climate Data Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ZipCodeStack API configuration
ZIPCODE_API_KEY = "zip_live_61Fw0aEgouDWn2ZO0sOk0aE0xnB68nWGv8mUwtsu"
ZIPCODE_BASE_URL = "https://app.zipcodestack.com/dashboard"

# Google Earth Engine Authentication
@st.cache_resource
def ee_authenticate():
    try:
        if "json_key" in st.secrets:
            st.info("Authenticating with Google Earth Engine using service account...")
            json_creds = st.secrets["json_key"]
            if isinstance(json_creds, (dict, st.runtime.secrets.AttrDict)):
                service_account_info = dict(json_creds)
            elif isinstance(json_creds, str):
                service_account_info = json.loads(json_creeds)
            else:
                raise ValueError("Invalid json_key format in secrets. Expected dict, AttrDict, or JSON string.")
            if "client_email" not in service_account_info:
                raise ValueError("Service account email address missing in json_key")
            creds = service_account.Credentials.from_service_account_info(
                service_account_info, scopes=['https://www.googleapis.com/auth/earthengine']
            )
            ee.Initialize(creds)
            st.success("Successfully authenticated with Google Earth Engine using service account.")
        else:
            st.info("Attempting authentication using Earth Engine CLI credentials...")
            ee.Initialize()
            st.success("Authenticated with Google Earth Engine using local CLI credentials.")
    except Exception as e:
        st.error(f"Failed to authenticate with Google Earth Engine: {str(e)}")
        st.markdown(
            "**Steps to resolve:**\n"
            "- **Local setup**: Create `.streamlit/secrets.toml` with a valid service account key, or run `earthengine authenticate`.\n"
            "  Example `secrets.toml`:\n"
            "  ```toml\n"
            "  [json_key]\n"
            "  type = 'service_account'\n"
            "  project_id = 'your-project-id'\n"
            "  private_key_id = 'your-private-key-id'\n"
            "  private_key = '-----BEGIN PRIVATE KEY-----\\n...\\n-----END PRIVATE KEY-----\\n'\n"
            "  client_email = 'your-service-account@your-project-id.iam.gserviceaccount.com'\n"
            "  client_id = 'your-client-id'\n"
            "  auth_uri = 'https://accounts.google.com/o/oauth2/auth'\n"
            "  token_uri = 'https://oauth2.googleapis.com/token'\n"
            "  auth_provider_x509_url = 'https://www.googleapis.com/oauth2/v1/certs'\n"
            "  client_x509_cert_url = 'https://www.googleapis.com/robot/v1/metadata/x509/your-service-account%40your-project-id.iam.gserviceaccount.com'\n"
            "  universe_domain = 'googleapis.com'\n"
            "  ```\n"
            "- **Cloud deployment**: Configure `[json_key]` in Streamlit secrets.\n"
            "- Ensure the service account has Earth Engine permissions (`roles/earthengine.user`).\n"
            "- Register at https://developers.google.com/earth-engine/guides/access.\n"
            "- Verify internet and Google Cloud project settings."
        )
        st.stop()

# Initialize Earth Engine
ee_authenticate()

# Initialize session state for drawn geometry
if 'drawn_geometry' not in st.session_state:
    st.session_state.drawn_geometry = None
if 'use_drawn_geometry' not in st.session_state:
    st.session_state.use_drawn_geometry = False

# Title and description
st.title("🌍 Enhanced Climate Data Dashboard")
st.markdown("**ERA5-Land and CHIRPS Comprehensive Analysis Platform**")

# ZipCodeStack API functions
@st.cache_data(ttl=3600)
def search_postal_codes(query, country_code=None):
    try:
        url = f"{ZIPCODE_BASE_URL}/search"
        params = {'apikey': ZIPCODE_API_KEY, 'codes': query}
        if country_code:
            params['country'] = country_code
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if 'results' in data and data['results']:
                return data['results']
        return []
    except Exception as e:
        st.error(f"Error searching postal codes: {str(e)}")
        return []

@st.cache_data(ttl=3600)
def get_postal_code_details(postal_code, country_code=None):
    try:
        url = f"{ZIPCODE_BASE_URL}/search"
        params = {'apikey': ZIPCODE_API_KEY, 'codes': postal_code}
        if country_code:
            params['country'] = country_code
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if 'results' in data and data['results']:
                result = data['results'][0]
                return {
                    'postal_code': result.get('postal_code', postal_code),
                    'city': result.get('city', 'Unknown'),
                    'state': result.get('state', 'Unknown'),
                    'country': result.get('country_code', 'Unknown'),
                    'latitude': float(result.get('latitude', 0)),
                    'longitude': float(result.get('longitude', 0))
                }
        return None
    except Exception as e:
        st.error(f"Error getting postal code details: {str(e)}")
        return None

# Sidebar configuration
st.sidebar.header("🌍 Area of Interest (AOI)")
aoi_method = st.sidebar.selectbox(
    "Select AOI Method",
    ["Global", "India", "Draw AOI", "Upload File", "Country Selection"]  # Removed "Postal/PIN Code"
)

# Initialize default geometry and map settings
geometry = ee.Geometry.Rectangle([-180, -90, 180, 90])
map_center = [20, 0]
zoom_start = 2

# Country boundaries
country_bounds = {
    "India": [68, 8, 98, 38],
    "USA": [-125, 25, -66, 49],
    "China": [73, 18, 135, 53],
    "Brazil": [-74, -34, -34, 5],
    "Australia": [113, -44, 154, -10],
    "Europe": [-10, 35, 30, 71],
    "Canada": [-141, 42, -52, 84],
    "Japan": [129, 31, 146, 46],
    "South Korea": [124, 33, 132, 39],
    "United Kingdom": [-8, 50, 2, 61]
}

# Country codes for postal code search
country_codes = {
    "United States": "US",
    "India": "IN",
    "United Kingdom": "GB",
    "Canada": "CA",
    "Australia": "AU",
    "Germany": "DE",
    "France": "FR",
    "Japan": "JP",
    "South Korea": "KR",
    "Brazil": "BR",
    "China": "CN",
    "Russia": "RU",
    "Italy": "IT",
    "Spain": "ES",
    "Netherlands": "NL",
    "Sweden": "SE",
    "Norway": "NO",
    "Denmark": "DK",
    "Finland": "FI",
    "Switzerland": "CH"
}

# AOI Selection Logic
def create_buffer_geometry(coords, buffer_km=50):
    point = ee.Geometry.Point(coords)
    return point.buffer(buffer_km * 1000)

if aoi_method == "India":
    geometry = ee.Geometry.Rectangle(country_bounds["India"])
    map_center = [20, 78]
    zoom_start = 5
    st.session_state.use_drawn_geometry = False

elif aoi_method == "Country Selection":
    country = st.sidebar.selectbox("Select Country/Region", list(country_bounds.keys()))
    bounds = country_bounds[country]
    geometry = ee.Geometry.Rectangle(bounds)
    map_center = [(bounds[1] + bounds[3])/2, (bounds[0] + bounds[2])/2]
    zoom_start = 4
    st.session_state.use_drawn_geometry = False

elif aoi_method == "Draw AOI":
    st.sidebar.info("🖊️ Use the drawing tools on the map below")
    st.sidebar.markdown("""
    **Instructions:**
    1. Use the polygon drawing tool on the map
    2. Draw your area of interest
    3. Click 'Apply Drawn AOI' button
    4. Run the analysis
    """)
    if st.session_state.use_drawn_geometry and st.session_state.drawn_geometry:
        try:
            geometry = ee.Geometry(st.session_state.drawn_geometry)
            centroid = geometry.centroid().coordinates().getInfo()
            map_center = [centroid[1], centroid[0]]
            zoom_start = 8
            st.sidebar.success("✅ Using drawn AOI")
        except:
            st.sidebar.warning("⚠️ Error with drawn geometry, using global extent")

elif aoi_method == "Upload File":
    st.sidebar.markdown("📁 **Upload KML/KMZ/Shapefile**")
    uploaded_file = st.sidebar.file_uploader(
        "Choose file",
        type=["kml", "kmz", "zip", "shp"],
        help="Support: KML, KMZ, or ZIP containing shapefile"
    )
    if uploaded_file:
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                file_path = os.path.join(tmpdir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                if uploaded_file.name.endswith(".kmz"):
                    with zipfile.ZipFile(file_path, 'r') as kmz:
                        kmz.extractall(tmpdir)
                        kml_files = [f for f in kmz.namelist() if f.endswith(".kml")]
                        if kml_files:
                            file_path = os.path.join(tmpdir, kml_files[0])
                elif uploaded_file.name.endswith(".zip"):
                    with zipfile.ZipFile(file_path, 'r') as zip_ref:
                        zip_ref.extractall(tmpdir)
                        shp_files = [f for f in zip_ref.namelist() if f.endswith(".shp")]
                        if shp_files:
                            file_path = os.path.join(tmpdir, shp_files[0])
                gdf = gpd.read_file(file_path)
                if not gdf.empty and gdf.geometry.iloc[0] is not None:
                    geom = gdf.geometry.iloc[0]
                    if geom.geom_type == "Polygon":
                        coords = [list(geom.exterior.coords)]
                        geometry = ee.Geometry.Polygon(coords)
                    elif geom.geom_type == "MultiPolygon":
                        coords = [list(poly.exterior.coords) for poly in geom.geoms]
                        geometry = ee.Geometry.MultiPolygon(coords)
                    centroid = geometry.centroid().coordinates().getInfo()
                    map_center = [centroid[1], centroid[0]]
                    zoom_start = 8
                    st.sidebar.success("✅ File uploaded successfully!")
                    st.session_state.use_drawn_geometry = False
        except Exception as e:
            st.sidebar.error(f"❌ Error processing file: {str(e)}")

# Date selection with presets
st.sidebar.header("📅 Date Range Selection")
date_presets = {
    "Last Year": (datetime.now() - timedelta(days=365), datetime.now() - timedelta(days=1)),
    "Last 6 Months": (datetime.now() - timedelta(days=180), datetime.now() - timedelta(days=1)),
    "Last 3 Months": (datetime.now() - timedelta(days=90), datetime.now() - timedelta(days=1)),
    "2023": (datetime(2023, 1, 1), datetime(2023, 12, 31)),
    "2022": (datetime(2022, 1, 1), datetime(2022, 12, 31)),
    "2021": (datetime(2021, 1, 1), datetime(2021, 12, 31)),
    "Custom": None
}
preset = st.sidebar.selectbox("Quick Date Selection", list(date_presets.keys()))
if preset == "Custom" or date_presets[preset] is None:
    start_date = st.sidebar.date_input(
        "Start Date",
        value=datetime(2023, 1, 1),
        min_value=datetime(1981, 1, 1),
        max_value=datetime.now() - timedelta(days=1)
    )
    end_date = st.sidebar.date_input(
        "End Date",
        value=datetime(2023, 12, 31),
        min_value=start_date,
        max_value=datetime.now() - timedelta(days=1)
    )
else:
    start_date, end_date = date_presets[preset]
    st.sidebar.info(f"📅 Selected: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

# Analysis options
st.sidebar.header("🔧 Analysis Options")
analysis_type = st.sidebar.multiselect(
    "Select Analysis Types",
    ["Time Series", "Statistics", "Trend Analysis", "Correlation Analysis"],  # Removed "Anomaly Detection", "Seasonal Analysis"
    default=["Time Series", "Statistics"]
)
temporal_aggregation = st.sidebar.selectbox(
    "Temporal Aggregation",
    ["Daily", "Weekly", "Monthly", "Seasonal", "Annual"]
)

# Convert dates to strings
start_date_str = start_date.strftime('%Y-%m-%d')
end_date_str = end_date.strftime('%Y-%m-%d')

# Data loading function
def load_datasets(start_date_str, end_date_str):
    try:
        era5_land = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR').filterDate(start_date_str, end_date_str)
        chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY').filterDate(start_date_str, end_date_str)
        return era5_land, chirps
    except Exception as e:
        st.error(f"Failed to load datasets: {str(e)}")
        return None, None

# Load datasets
with st.spinner("🔄 Loading datasets..."):
    era5_land, chirps = load_datasets(start_date_str, end_date_str)

if era5_land is None or chirps is None:
    st.error("Failed to load datasets. Please check your connection and try again.")
    st.stop()

# Available bands
era5_bands = {
    'temperature_2m_max': 'Maximum Temperature (2m)',
    'temperature_2m_min': 'Minimum Temperature (2m)',
    'temperature_2m': 'Temperature (2m)',
    'dewpoint_temperature_2m': 'Dewpoint Temperature (2m)',
    'total_precipitation_sum': 'Total Precipitation',
    'surface_pressure': 'Surface Pressure',
    'u_component_of_wind_10m': 'Wind U-component (10m)',
    'v_component_of_wind_10m': 'Wind V-component (10m)',
    'surface_solar_radiation_downwards_sum': 'Solar Radiation',
    'surface_thermal_radiation_downwards_sum': 'Thermal Radiation',
    'total_evaporation_sum': 'Total Evaporation',
    'potential_evaporation_sum': 'Potential Evaporation',
    'runoff_sum': 'Runoff',
    'sub_surface_runoff_sum': 'Sub-surface Runoff'
}
chirps_bands = {'precipitation': 'CHIRPS Precipitation'}

# Band selection
selected_era5_bands = st.sidebar.multiselect(
    "Select ERA5-Land Variables",
    list(era5_bands.keys()),
    default=['temperature_2m_max', 'temperature_2m_min', 'total_precipitation_sum'],
    format_func=lambda x: era5_bands[x]
)
include_chirps = st.sidebar.checkbox("Include CHIRPS Precipitation", value=True)

# Visualization parameters
vis_params = {
    'temperature_2m_max': {'min': 250, 'max': 320, 'palette': ['#000080', '#0000d9', '#4000ff', '#8000ff', '#0080ff', '#00ffff', '#00ff80', '#80ff00', '#daff00', '#ffff00', '#fff500', '#ffda00', '#ffb000', '#ffa400', '#ff4f00', '#ff2500', '#ff0a00', '#ff00ff']},
    'temperature_2m_min': {'min': 230, 'max': 300, 'palette': ['#000080', '#0000d9', '#4000ff', '#8000ff', '#0080ff', '#00ffff', '#00ff80', '#80ff00', '#daff00', '#ffff00', '#fff500', '#ffda00', '#ffb000', '#ffa400', '#ff4f00', '#ff2500', '#ff0a00', '#ff00ff']},
    'temperature_2m': {'min': 240, 'max': 310, 'palette': ['#000080', '#0000d9', '#4000ff', '#8000ff', '#0080ff', '#00ffff', '#00ff80', '#80ff00', '#daff00', '#ffff00', '#fff500', '#ffda00', '#ffb000', '#ffa400', '#ff4f00', '#ff2500', '#ff0a00', '#ff00ff']},
    'total_precipitation_sum': {'min': 0, 'max': 0.1, 'palette': ['#ffffff', '#00ffff', '#0080ff', '#da00ff', '#ffa400', '#ff0000']},
    'precipitation': {'min': 0, 'max': 50, 'palette': ['#ffffff', '#00ffff', '#0080ff', '#da00ff', '#ffa400', '#ff0000']},
    'surface_pressure': {'min': 95000, 'max': 105000, 'palette': ['#0000ff', '#00ff00', '#ffff00', '#ff0000']},  # blue, green, yellow, red
    'dewpoint_temperature_2m': {'min': 230, 'max': 300, 'palette': ['#800080', '#0000ff', '#00ff00', '#ffff00', '#ff0000']},  # purple, blue, green, yellow, red
    'u_component_of_wind_10m': {'min': -20, 'max': 20, 'palette': ['#ff0000', '#ffffff', '#0000ff']},  # red, white, blue
    'v_component_of_wind_10m': {'min': -20, 'max': 20, 'palette': ['#ff0000', '#ffffff', '#0000ff']},  # red, white, blue
    'surface_solar_radiation_downwards_sum': {'min': 0, 'max': 30000000, 'palette': ['#000000', '#0000ff', '#800080', '#00ffff', '#00ff00', '#ffff00', '#ff0000']},  # black, blue, purple, cyan, green, yellow, red
    'total_evaporation_sum': {'min': -0.01, 'max': 0.01, 'palette': ['#a52a2a', '#ffff00', '#00ff00', '#0000ff']},  # brown, yellow, green, blue
    'runoff_sum': {'min': 0, 'max': 0.001, 'palette': ['#ffffff', '#0000ff', '#00ff00', '#ff0000']}  # white, blue, green, red
}

# Data extraction function
def extract_time_series_data(collection, band, geometry_json, start_date, end_date):
    try:
        geometry = ee.Geometry(geometry_json)
        def extract_values(image):
            stats = image.select(band).reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry,
                scale=1000,
                maxPixels=1e9
            )
            return ee.Feature(None, {
                'date': ee.Date(image.get('system:time_start')).format('YYYY-MM-dd'),
                band: stats.get(band)
            })
        features = collection.map(extract_values)
        data = features.getInfo()
        rows = []
        for feature in data['features']:
            props = feature['properties']
            if props[band] is not None:
                rows.append({
                    'date': pd.to_datetime(props['date']),
                    'value': props[band]
                })
        if rows:
            df = pd.DataFrame(rows)
            df.set_index('date', inplace=True)
            return df.sort_index()
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error extracting data for {band}: {str(e)}")
        return pd.DataFrame()

def aggregate_time_series(df, aggregation):
    """Aggregate time series DataFrame according to the selected temporal aggregation."""
    if df.empty:
        return df
    if aggregation == "Daily":
        return df  # Already daily
    elif aggregation == "Weekly":
        return df.resample('W').mean()
    elif aggregation == "Monthly":
        return df.resample('M').mean()
    elif aggregation == "Seasonal":
        # Define seasons as DJF, MAM, JJA, SON
        df = df.copy()
        df['season'] = ((df.index.month % 12 + 3) // 3)
        season_map = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'}
        df['season_name'] = df['season'].map(season_map)
        return df.groupby([df.index.year, 'season_name'])['value'].mean().unstack()
    elif aggregation == "Annual":
        return df.resample('Y').mean()
    else:
        return df

# Convert geometry to JSON for processing
geometry_json = geometry.getInfo()

# Create map for AOI visualization and drawing
st.header("🗺️ Area of Interest")
m = folium.Map(location=map_center, zoom_start=zoom_start)

# Add drawing tools for Draw AOI method
if aoi_method == "Draw AOI":
    from folium.plugins import Draw
    draw = Draw(
        export=True,
        filename='data.geojson',
        position='topleft',
        draw_options={
            'polyline': False,
            'rectangle': True,
            'polygon': True,
            'circle': False,
            'marker': False,
            'circlemarker': False,
        },
        edit_options={'edit': False}
    )
    draw.add_to(m)

# Add current AOI boundary if not drawing
if aoi_method != "Draw AOI" or st.session_state.use_drawn_geometry:
    try:
        if geometry_json['type'] == 'Polygon':
            coords = geometry_json['coordinates'][0]
            folium.Polygon(
                locations=[[lat, lon] for lon, lat in coords],
                color='red',
                weight=2,
                fill=False,
                popup='Area of Interest'
            ).add_to(m)
        elif geometry_json['type'] == 'Rectangle':
            bounds = geometry_json['coordinates'][0]
            folium.Rectangle(
                bounds=[[bounds[0][1], bounds[0][0]], [bounds[2][1], bounds[2][0]]],
                color='red',
                weight=2,
                fill=False,
                popup='Area of Interest'
            ).add_to(m)
    except Exception as e:
        st.warning(f"Could not display AOI boundary: {str(e)}")

# Display map
map_data = st_folium(m, width=700, height=500, returned_objects=["last_object_clicked_popup", "all_drawings"])

# Handle drawn AOI
if aoi_method == "Draw AOI":
    if map_data.get('all_drawings') and len(map_data['all_drawings']) > 0:
        st.info("🖊️ AOI drawn on map. Click 'Apply Drawn AOI' to use it.")
        if st.button("✅ Apply Drawn AOI", type="primary"):
            try:
                latest_drawing = map_data['all_drawings'][-1]
                drawn_geom = latest_drawing['geometry']
                if drawn_geom['type'] == 'Polygon':
                    coords = drawn_geom['coordinates']
                    st.session_state.drawn_geometry = {'type': 'Polygon', 'coordinates': coords}
                    st.session_state.use_drawn_geometry = True
                    st.success("✅ AOI updated! You can now run the analysis.")
                    st.rerun()
                else:
                    st.error("❌ Please draw a polygon area")
            except Exception as e:
                st.error(f"❌ Error processing drawn AOI: {str(e)}")
    else:
        st.info("ℹ️ Draw a polygon on the map to define your area of interest")

# Main analysis
if st.button("🚀 Run Analysis", type="primary", key="main_analysis"):
    with st.spinner("🔄 Processing data..."):
        if st.session_state.use_drawn_geometry and st.session_state.drawn_geometry:
            geometry_json = st.session_state.drawn_geometry
        data_dict = {}
        for band in selected_era5_bands:
            df = extract_time_series_data(era5_land, band, geometry_json, start_date_str, end_date_str)
            df = aggregate_time_series(df, temporal_aggregation)
            if not df.empty:
                data_dict[band] = df
        if include_chirps:
            df = extract_time_series_data(chirps, 'precipitation', geometry_json, start_date_str, end_date_str)
            if not df.empty:
                data_dict['chirps_precipitation'] = df
    if not data_dict:
        st.error("❌ No data extracted. Please check your AOI and date range.")
        st.stop()
    st.session_state['data_dict'] = data_dict
    st.session_state['analysis_complete'] = True

# Analysis results
if st.session_state.get('analysis_complete', False) and 'data_dict' in st.session_state:
    data_dict = st.session_state['data_dict']
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Time Series", "📈 Statistics", "🔍 Trend Analysis", "🔗 Correlations"])
    
    with tab1:
        st.header("📊 Time Series Analysis")
        temp_data = {}
        for band in ['temperature_2m_max', 'temperature_2m_min', 'temperature_2m']:
            if band in data_dict:
                temp_data[band] = data_dict[band]['value'] - 273.15
        if temp_data:
            fig = go.Figure()
            colors = ['red', 'blue', 'green']
            for i, (band, data) in enumerate(temp_data.items()):
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data.values,
                    mode='lines',
                    name=era5_bands[band],
                    line=dict(color=colors[i % len(colors)])
                ))
            fig.update_layout(
                title="Temperature Time Series",
                xaxis_title="Date",
                yaxis_title="Temperature (°C)",
                hovermode='x unified',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        
        precip_data = {}
        # --- AGGREGATE ACCORDING TO TEMPORAL AGGREGATION ---
        if 'total_precipitation_sum' in data_dict:
            era5_precip = data_dict['total_precipitation_sum'].copy()
            era5_precip_agg = aggregate_time_series(era5_precip, temporal_aggregation)
            precip_data['ERA5'] = era5_precip_agg['value'] * 1000  # Convert to mm
        if 'chirps_precipitation' in data_dict:
            chirps_precip = data_dict['chirps_precipitation'].copy()
            chirps_precip_agg = aggregate_time_series(chirps_precip, temporal_aggregation)
            precip_data['CHIRPS'] = chirps_precip_agg['value']
        if precip_data:
            fig = go.Figure()
            colors = ['blue', 'orange']
            for i, (source, data) in enumerate(precip_data.items()):
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data.values,
                    mode='lines',
                    name=f"{source} Precipitation",
                    line=dict(color=colors[i])
                ))
            fig.update_layout(
                title=f"Precipitation Comparison ({temporal_aggregation})",
                xaxis_title="Date",
                yaxis_title="Precipitation (mm)",
                hovermode='x unified',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("📈 Statistical Summary")
        stats_data = []
        for band, df in data_dict.items():
            values = df['value'].values
            if 'temperature' in band:
                values = values - 273.15
                unit = "°C"
            elif 'precipitation' in band and band != 'chirps_precipitation':
                values = values * 1000
                unit = "mm"
            elif band == 'chirps_precipitation':
                unit = "mm"
            else:
                unit = "units"
            stats = {
                'Variable': era5_bands.get(band, band.replace('_', ' ').title()),
                'Count': len(values),
                'Mean': f"{np.mean(values):.2f} {unit}",
                'Std Dev': f"{np.std(values):.2f} {unit}",
                'Min': f"{np.min(values):.2f} {unit}",
                'Max': f"{np.max(values):.2f} {unit}",
                'Range': f"{np.max(values) - np.min(values):.2f} {unit}",
                'Skewness': f"{skew(values):.2f}",
                'Kurtosis': f"{kurtosis(values):.2f}",
                'CV': f"{np.std(values)/np.mean(values)*100:.1f}%"
            }
            stats_data.append(stats)
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True)
        
        st.subheader("📊 Data Distributions")
        n_vars = len(data_dict)
        cols = min(3, n_vars)
        rows = (n_vars - 1) // cols + 1
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[era5_bands.get(band, band) for band in data_dict.keys()]
        )
        for i, (band, df) in enumerate(data_dict.items()):
            row = i // cols + 1
            col = i % cols + 1
            values = df['value'].values
            if 'temperature' in band:
                values = values - 273.15
            elif 'precipitation' in band and band != 'chirps_precipitation':
                values = values * 1000
            fig.add_trace(
                go.Histogram(x=values, name=band, showlegend=False),
                row=row, col=col
            )
        fig.update_layout(height=300*rows, title_text="Variable Distributions")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("🔍 Trend Analysis")
        trend_data = []
        for band, df in data_dict.items():
            values = df['value'].values
            dates = pd.to_datetime(df.index)
            time_numeric = (dates - dates[0]).days
            if 'temperature' in band:
                values = values - 273.15
                unit = "°C"
            elif 'precipitation' in band and band != 'chirps_precipitation':
                values = values * 1000
                unit = "mm"
            elif band == 'chirps_precipitation':
                unit = "mm"
            else:
                unit = "units"
            slope, intercept, r_value, p_value, std_err = linregress(time_numeric, values)
            slope_per_year = slope * 365.25
            trend_data.append({
                'Variable': era5_bands.get(band, band.replace('_', ' ').title()),
                'Trend (per year)': f"{slope_per_year:.4f} {unit}/year",
                'R-squared': f"{r_value**2:.3f}",
                'P-value': f"{p_value:.4f}",
                'Significance': "Yes" if p_value < 0.05 else "No"
            })
        trend_df = pd.DataFrame(trend_data)
        st.dataframe(trend_df, use_container_width=True)
        
        st.subheader("📈 Trend Visualization")
        selected_var = st.selectbox(
            "Select variable for trend plot",
            list(data_dict.keys()),
            format_func=lambda x: era5_bands.get(x, x.replace('_', ' ').title()),
            key="trend_var_select"
        )
        if selected_var:
            df = data_dict[selected_var]
            values = df['value'].values
            dates = df.index
            if 'temperature' in selected_var:
                values = values - 273.15
                unit = "°C"
            elif 'precipitation' in selected_var and selected_var != 'chirps_precipitation':
                values = values * 1000
                unit = "mm"
            elif selected_var == 'chirps_precipitation':
                unit = "mm"
            else:
                unit = "units"
            time_numeric = (pd.to_datetime(dates) - pd.to_datetime(dates[0])).days
            slope, intercept, _, _, _ = linregress(time_numeric, values)
            trend_line = slope * time_numeric + intercept
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates, y=values,
                mode='lines',
                name='Data',
                line=dict(color='blue', width=1)
            ))
            fig.add_trace(go.Scatter(
                x=dates, y=trend_line,
                mode='lines',
                name='Trend',
                line=dict(color='red', width=2, dash='dash')
            ))
            fig.update_layout(
                title=f"Trend Analysis: {era5_bands.get(selected_var, selected_var)}",
                xaxis_title="Date",
                yaxis_title=f"Value ({unit})",
                hovermode='x unified',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        
        if not trend_df.empty:
            csv = trend_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Trend Analysis CSV",
                data=csv,
                file_name=f"trend_{start_date_str}_{end_date_str}.csv",
                mime="text/csv"
            )
    
    with tab4:
        st.header("🔗 Correlation Analysis")
        if len(data_dict) >= 2:
            correlation_data = {}
            for band, df in data_dict.items():
                values = df['value'].values
                if 'temperature' in band:
                    values = values - 273.15
                elif 'precipitation' in band and band != 'chirps_precipitation':
                    values = values * 1000
                correlation_data[band] = values
            common_dates = None
            for band, df in data_dict.items():
                if common_dates is None:
                    common_dates = set(df.index)
                else:
                    common_dates = common_dates.intersection(set(df.index))
            if common_dates:
                aligned_data = {}
                for band in correlation_data:
                    df = data_dict[band]
                    aligned_data[band] = df.loc[list(common_dates)]['value'].values
                    if 'temperature' in band:
                        aligned_data[band] = aligned_data[band] - 273.15
                    elif 'precipitation' in band and band != 'chirps_precipitation':
                        aligned_data[band] = aligned_data[band] * 1000
                corr_df = pd.DataFrame(aligned_data)
                correlation_matrix = corr_df.corr()
                fig = px.imshow(
                    correlation_matrix,
                    labels=dict(color="Correlation"),
                    x=[era5_bands.get(col, col) for col in correlation_matrix.columns],
                    y=[era5_bands.get(row, row) for row in correlation_matrix.index],
                    color_continuous_scale='RdBu_r',
                    aspect="auto"
                )
                fig.update_layout(title="Correlation Matrix", height=500)
                st.plotly_chart(fig, use_container_width=True)
                st.subheader("Correlation Coefficients")
                st.dataframe(correlation_matrix, use_container_width=True)
        else:
            st.info("Need at least 2 variables for correlation analysis")
    
# =========================
# Advanced Analysis Section
# =========================
if st.session_state.get('analysis_complete', False):
    st.header("🔬 Advanced Analysis")
    
    # Remove Anomaly Detection and Seasonal Analysis tabs
    adv_tab1, adv_tab2, adv_tab3, adv_tab4 = st.tabs([
        "🎯 Extreme Events", 
        "📊 Forecasting", 
        "🌡️ Climate Indices", 
        "📈 Change Point Detection"
    ])
    
    with adv_tab1:
        with st.expander("ℹ️ What is Extreme Events Analysis?"):
            st.markdown("""
            **What is this?**  
            Extreme events analysis identifies unusually high or low values in climate variables, such as heatwaves or heavy rainfall events.

            **Why we use it?**  
            Detecting extremes helps in risk assessment, disaster preparedness, and understanding climate variability and change.

            **Interpretation:**  
            - High extremes: Values above a chosen percentile (e.g., 95th) are considered unusually high.
            - Low extremes: Values below a chosen percentile (e.g., 5th) are unusually low.
            - The number and frequency of extremes can indicate climate risks.

            **References:**  
            - IPCC (2021). [Climate Change 2021: The Physical Science Basis](https://www.ipcc.ch/report/ar6/wg1/)
            - WMO (2018). [Guidelines on the Definition and Monitoring of Extreme Weather and Climate Events](https://library.wmo.int/doc_num.php?explnum_id=5445)
            """)
        st.subheader("🎯 Extreme Events Analysis")
        extreme_var = st.selectbox(
            "Select variable for extreme events",
            list(data_dict.keys()),
            format_func=lambda x: era5_bands.get(x, x.replace('_', ' ').title()),
            key="extreme_var_select"
        )
        
        if extreme_var:
            df = data_dict[extreme_var].copy()
            values = df['value'].values
            if 'temperature' in extreme_var:
                values = values - 273.15
                unit = "°C"
            elif 'precipitation' in extreme_var and extreme_var != 'chirps_precipitation':
                values = values * 1000
                unit = "mm"
            elif extreme_var == 'chirps_precipitation':
                unit = "mm"
            else:
                unit = "units"
            
            col1, col2 = st.columns(2)
            with col1:
                percentile_high = st.slider("High Extreme Percentile", 90, 99, 95)
            with col2:
                percentile_low = st.slider("Low Extreme Percentile", 1, 10, 5)
            
            high_threshold = np.percentile(values, percentile_high)
            low_threshold = np.percentile(values, percentile_low)
            
            high_extremes = values >= high_threshold
            low_extremes = values <= low_threshold
            
            # Extreme events plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df.index, y=values,
                mode='lines',
                name='Data',
                line=dict(color='blue', width=1)
            ))
            fig.add_trace(go.Scatter(
                x=df.index[high_extremes], y=values[high_extremes],
                mode='markers',
                name=f'High Extremes (>{percentile_high}%)',
                marker=dict(color='red', size=8)
            ))
            fig.add_trace(go.Scatter(
                x=df.index[low_extremes], y=values[low_extremes],
                mode='markers',
                name=f'Low Extremes (<{percentile_low}%)',
                marker=dict(color='purple', size=8)
            ))
            fig.add_hline(y=high_threshold, line_dash="dash", line_color="red", 
                         annotation_text=f"{percentile_high}th percentile")
            fig.add_hline(y=low_threshold, line_dash="dash", line_color="purple", 
                         annotation_text=f"{percentile_low}th percentile")
            fig.update_layout(
                title=f"Extreme Events: {era5_bands.get(extreme_var, extreme_var)}",
                xaxis_title="Date",
                yaxis_title=f"Value ({unit})",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Extreme events statistics
            n_high = np.sum(high_extremes)
            n_low = np.sum(low_extremes)
            st.write(f"**📊 Extreme Events Summary:**")
            st.write(f"- High extremes: {n_high} events ({n_high/len(values)*100:.1f}%)")
            st.write(f"- Low extremes: {n_low} events ({n_low/len(values)*100:.1f}%)")
            st.write(f"- High threshold: {high_threshold:.2f} {unit}")
            st.write(f"- Low threshold: {low_threshold:.2f} {unit}")
            
            # Export extreme events data
            extreme_df = pd.DataFrame({
                'date': df.index,
                'value': values,
                'high_extreme': high_extremes,
                'low_extreme': low_extremes
            })
            csv = extreme_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Extreme Events CSV",
                data=csv,
                file_name=f"extreme_events_{extreme_var}_{start_date_str}_{end_date_str}.csv",
                mime="text/csv"
            )
    
    with adv_tab2:
        with st.expander("ℹ️ What is Simple Forecasting?"):
            st.markdown("""
            **What is this?**  
            Simple forecasting uses historical data trends (e.g., moving averages) to predict future values.

            **Why we use it?**  
            Forecasting helps anticipate future climate conditions for planning and management.

            **Interpretation:**  
            - The forecast shows expected values if current trends continue.
            - Simple methods do not account for seasonality or external factors.

            **References:**  
            - Hyndman, R.J., & Athanasopoulos, G. (2021). [Forecasting: Principles and Practice](https://otexts.com/fpp3/)
            """)
        st.subheader("📊 Simple Forecasting")
        forecast_var = st.selectbox(
            "Select variable for forecasting",
            list(data_dict.keys()),
            format_func=lambda x: era5_bands.get(x, x.replace('_', ' ').title()),
            key="forecast_var_select"
        )
        
        if forecast_var:
            df = data_dict[forecast_var].copy()
            values = df['value'].values
            if 'temperature' in forecast_var:
                values = values - 273.15
                unit = "°C"
            elif 'precipitation' in forecast_var and forecast_var != 'chirps_precipitation':
                values = values * 1000
                unit = "mm"
            elif forecast_var == 'chirps_precipitation':
                unit = "mm"
            else:
                unit = "units"
            
            # Simple moving average forecast
            window_size = st.slider("Moving Average Window (days)", 7, 60, 30)
            forecast_days = st.slider("Forecast Period (days)", 7, 90, 30)
            
            # Calculate moving average
            moving_avg = pd.Series(values).rolling(window=window_size).mean()
            last_avg = moving_avg.iloc[-1]
            
            # Create forecast dates
            last_date = df.index[-1]
            forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                         periods=forecast_days, freq='D')
            
            # Simple forecast (using last moving average as constant forecast)
            forecast_values = [last_avg] * forecast_days
            
            # Plot historical data and forecast
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df.index, y=values,
                mode='lines',
                name='Historical Data',
                line=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                x=df.index, y=moving_avg,
                mode='lines',
                name=f'{window_size}-day Moving Average',
                line=dict(color='orange')
            ))
            fig.add_trace(go.Scatter(
                x=forecast_dates, y=forecast_values,
                mode='lines',
                name='Simple Forecast',
                line=dict(color='red', dash='dash')
            ))
            fig.update_layout(
                title=f"Simple Forecast: {era5_bands.get(forecast_var, forecast_var)}",
                xaxis_title="Date",
                yaxis_title=f"Value ({unit})",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.info(f"📈 Simple forecast value: {last_avg:.2f} {unit} (constant for {forecast_days} days)")
            st.warning("⚠️ This is a very basic forecast. For accurate predictions, use specialized forecasting models.")
    
    with adv_tab3:
        with st.expander("ℹ️ What are Climate Indices?"):
            st.markdown("""
            **What is this?**  
            Climate indices are derived metrics (e.g., Growing Degree Days, Heating/Cooling Degree Days) that summarize climate impacts on agriculture, health, and infrastructure.

            **Why we use it?**  
            Indices help translate raw climate data into actionable information for specific applications.

            **Interpretation:**  
            - GDD: Indicates crop growth potential.
            - Heating/Cooling Degree Days: Estimate energy demand for heating/cooling.
            - Dry/Heavy Rain Days: Assess drought or flood risk.

            **References:**  
            - FAO (2017). [Agroclimatic Indices and Yield Forecasting](https://www.fao.org/3/x0490e/x0490e07.htm)
            - NOAA. [Climate Indices: Definitions and Uses](https://www.ncdc.noaa.gov/teleconnections/)
            """)
        st.subheader("🌡️ Climate Indices Calculator")
        
        # Check if we have temperature data
        temp_vars = [k for k in data_dict.keys() if 'temperature' in k]
        precip_vars = [k for k in data_dict.keys() if 'precipitation' in k]
        
        if temp_vars:
            st.write("**🌡️ Temperature-based Indices**")
            temp_var = st.selectbox("Select temperature variable", temp_vars, 
                                   format_func=lambda x: era5_bands.get(x, x))
            
            df_temp = data_dict[temp_var].copy()
            temp_values = df_temp['value'].values - 273.15  # Convert to Celsius
            
            # Growing Degree Days (GDD)
            base_temp = st.slider("Base Temperature for GDD (°C)", 0, 20, 10)
            gdd = np.maximum(temp_values - base_temp, 0)
            cumulative_gdd = np.cumsum(gdd)
            
            # Heating/Cooling Degree Days
            comfort_temp = st.slider("Comfort Temperature (°C)", 15, 25, 18)
            heating_dd = np.maximum(comfort_temp - temp_values, 0)
            cooling_dd = np.maximum(temp_values - comfort_temp, 0)
            
            # Plot climate indices
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    'Growing Degree Days (Daily)',
                    'Cumulative Growing Degree Days',
                    'Heating Degree Days',
                    'Cooling Degree Days'
                ]
            )
            
            fig.add_trace(go.Scatter(x=df_temp.index, y=gdd, name='Daily GDD'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_temp.index, y=cumulative_gdd, name='Cumulative GDD'), row=1, col=2)
            fig.add_trace(go.Scatter(x=df_temp.index, y=heating_dd, name='Heating DD'), row=2, col=1)
            fig.add_trace(go.Scatter(x=df_temp.index, y=cooling_dd, name='Cooling DD'), row=2, col=2)
            
            fig.update_layout(height=600, title_text="Climate Indices", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary statistics
            st.write("**📊 Climate Indices Summary:**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total GDD", f"{cumulative_gdd[-1]:.0f}")
            with col2:
                st.metric("Avg Daily GDD", f"{np.mean(gdd):.1f}")
            with col3:
                st.metric("Total Heating DD", f"{np.sum(heating_dd):.0f}")
            with col4:
                st.metric("Total Cooling DD", f"{np.sum(cooling_dd):.0f}")
        
        if precip_vars:
            st.write("**🌧️ Precipitation-based Indices**")
            precip_var = st.selectbox("Select precipitation variable", precip_vars,
                                    format_func=lambda x: era5_bands.get(x, x.replace('_', ' ').title()))
            
            df_precip = data_dict[precip_var].copy()
            precip_values = df_precip['value'].values
            if precip_var != 'chirps_precipitation':
                precip_values = precip_values * 1000  # Convert to mm
            
            # Precipitation indices
            dry_threshold = st.slider("Dry Day Threshold (mm)", 0.1, 5.0, 1.0)
            heavy_threshold = st.slider("Heavy Rain Threshold (mm)", 10, 50, 20)
            
            dry_days = precip_values < dry_threshold
            wet_days = precip_values >= dry_threshold
            heavy_days = precip_values >= heavy_threshold
            
            # Calculate consecutive dry/wet days
            dry_spells = []
            wet_spells = []
            current_dry = 0
            current_wet = 0
            
            for is_dry in dry_days:
                if is_dry:
                    current_dry += 1
                    if current_wet > 0:
                        wet_spells.append(current_wet)
                        current_wet = 0
                else:
                    current_wet += 1
                    if current_dry > 0:
                        dry_spells.append(current_dry)
                        current_dry = 0
            
            # Add final spell
            if current_dry > 0:
                dry_spells.append(current_dry)
            if current_wet > 0:
                wet_spells.append(current_wet)
            
            # Display precipitation indices
            st.write("**📊 Precipitation Indices Summary:**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Dry Days", f"{np.sum(dry_days)}")
            with col2:
                st.metric("Wet Days", f"{np.sum(wet_days)}")
            with col3:
                st.metric("Heavy Rain Days", f"{np.sum(heavy_days)}")
            with col4:
                st.metric("Max Dry Spell", f"{max(dry_spells) if dry_spells else 0} days")
    
    with adv_tab4:
        with st.expander("ℹ️ What is Change Point Detection?"):
            st.markdown("""
            **What is this?**  
            Change point detection identifies times when the statistical properties of a time series change, such as a shift in mean or trend.

            **Why we use it?**  
            Detecting change points helps reveal regime shifts, impacts of interventions, or climate change signals.

            **Interpretation:**  
            - Detected points indicate potential abrupt changes in the data.
            - Review detected dates for correspondence with known events or changes.

            **References:**  
            - Killick, R., & Eckley, I.A. (2014). [changepoint: An R Package for Change Point Analysis](https://www.jstatsoft.org/article/view/v058i03)
            - Reeves, J. et al. (2007). [A Review and Comparison of Change Point Detection Techniques](https://journals.ametsoc.org/view/journals/apme/46/6/jam2493.1.xml)
            """)
        st.subheader("📈 Change Point Detection")
        changepoint_var = st.selectbox(
            "Select variable for change point detection",
            list(data_dict.keys()),
            format_func=lambda x: era5_bands.get(x, x.replace('_', ' ').title()),
            key="changepoint_var_select"
        )
        
        if changepoint_var:
            df = data_dict[changepoint_var].copy()
            values = df['value'].values
            if 'temperature' in changepoint_var:
                values = values - 273.15
                unit = "°C"
            elif 'precipitation' in changepoint_var and changepoint_var != 'chirps_precipitation':
                values = values * 1000
                unit = "mm"
            elif changepoint_var == 'chirps_precipitation':
                unit = "mm"
            else:
                unit = "units"
            
            # Simple change point detection using CUSUM
            def cusum_changepoint(data, threshold_factor=2):
                mean_val = np.mean(data)
                cusum_pos = np.zeros(len(data))
                cusum_neg = np.zeros(len(data))
                
                for i in range(1, len(data)):
                    cusum_pos[i] = max(0, cusum_pos[i-1] + data[i] - mean_val)
                    cusum_neg[i] = min(0, cusum_neg[i-1] + data[i] - mean_val)
                
                threshold = threshold_factor * np.std(data)
                changepoints = []
                
                for i in range(len(data)):
                    if abs(cusum_pos[i]) > threshold or abs(cusum_neg[i]) > threshold:
                        changepoints.append(i)
                
                return changepoints, cusum_pos, cusum_neg
            
            threshold_factor = st.slider("Sensitivity (lower = more sensitive)", 1.0, 5.0, 2.0, 0.1)
            changepoints, cusum_pos, cusum_neg = cusum_changepoint(values, threshold_factor)
            
            # Plot change point detection
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=[
                    f'Time Series with Change Points: {era5_bands.get(changepoint_var, changepoint_var)}',
                    'CUSUM Statistics'
                ],
                vertical_spacing=0.1
            )
            
            # Original time series
            fig.add_trace(go.Scatter(
                x=df.index, y=values,
                mode='lines',
                name='Data',
                line=dict(color='blue')
            ), row=1, col=1)
            
            # Mark change points
            if changepoints:
                cp_dates = df.index[changepoints]
                cp_values = values[changepoints]
                fig.add_trace(go.Scatter(
                    x=cp_dates, y=cp_values,
                    mode='markers',
                    name='Change Points',
                    marker=dict(color='red', size=10, symbol='diamond')
                ), row=1, col=1)
            
            # CUSUM plot
            fig.add_trace(go.Scatter(
                x=df.index, y=cusum_pos,
                mode='lines',
                name='CUSUM+',
                line=dict(color='green')
            ), row=2, col=1)
            fig.add_trace(go.Scatter(
                x=df.index, y=cusum_neg,
                mode='lines',
                name='CUSUM-',
                line=dict(color='red')
            ), row=2, col=1)
            
            fig.update_yaxes(title_text=f"Value ({unit})", row=1, col=1)
            fig.update_yaxes(title_text="CUSUM", row=2, col=1)
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_layout(height=700, title_text="Change Point Detection Analysis")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Change point summary
            if changepoints:
                st.success(f"🔍 Detected {len(changepoints)} potential change points")
                change_dates = df.index[changepoints]
                st.write("**Change Point Dates:**")
                for i, date in enumerate(change_dates[:10]):  # Show first 10
                    st.write(f"- {date.strftime('%Y-%m-%d')}")
                if len(change_dates) > 10:
                    st.write(f"... and {len(change_dates) - 10} more")
            else:
                st.info("ℹ️ No significant change points detected. Try adjusting the sensitivity.")
            
            st.info("💡 Change points indicate potential shifts in the mean or trend of the time series.")

# =========================
# Map Visualization Section
# =========================
st.header("🗺️ Map Visualization")

with st.expander("Map Visualization Controls", expanded=True):
    st.markdown("**Visualize spatial patterns for selected variables and time periods.**")
    map_var = st.selectbox(
        "Select variable for map visualization",
        list(era5_bands.keys()) + (["precipitation"] if include_chirps else []),
        format_func=lambda x: era5_bands.get(x, chirps_bands.get(x, x.replace('_', ' ').title()))
    )
    map_stat = st.selectbox(
        "Select map type",
        ["Mean", "Max", "Min", "Sum", "Trend", "Threshold"]  # Added "Sum"
    )
    map_start = st.date_input(
        "Map Start Date",
        value=start_date,
        min_value=datetime(1981, 1, 1),
        max_value=end_date
    )
    map_end = st.date_input(
        "Map End Date",
        value=end_date,
        min_value=map_start,
        max_value=datetime.now() - timedelta(days=1)
    )
    if map_stat == "Threshold":
        threshold_val = st.number_input("Threshold value", value=30.0 if "temp" in map_var else 10.0)
        threshold_type = st.selectbox("Threshold type", ["Above", "Below"])

    show_map = st.button("🌐 Show Map", key="show_map_btn")

    # Use session state to persist map display
    if show_map:
        st.session_state['show_map_flag'] = True

# Only show map if flag is set
if st.session_state.get('show_map_flag', False):
    with st.spinner("Generating map..."):
        # Select collection and band
        if map_var == "precipitation":
            collection = chirps
            band = "precipitation"
            vis = vis_params["precipitation"]
        else:
            collection = era5_land
            band = map_var
            vis = vis_params.get(band, {})
        # Filter by date
        map_start_str = map_start.strftime('%Y-%m-%d')
        map_end_str = map_end.strftime('%Y-%m-%d')
        col = collection.filterDate(map_start_str, map_end_str)
        # Reduce images
        if map_stat == "Mean":
            img = col.select(band).mean()
            legend_title = "Mean"
            # Convert temperature from Kelvin to Celsius for visualization and legend
            if band in ["temperature_2m_max", "temperature_2m_min", "temperature_2m"]:
                img = img.subtract(273.15)
                vis = vis.copy()
                vis['min'] = vis['min'] - 273.15
                vis['max'] = vis['max'] - 273.15
        elif map_stat == "Max":
            img = col.select(band).max()
            legend_title = "Max"
            if band in ["temperature_2m_max", "temperature_2m_min", "temperature_2m"]:
                img = img.subtract(273.15)
                vis = vis.copy()
                vis['min'] = vis['min'] - 273.15
                vis['max'] = vis['max'] - 273.15
        elif map_stat == "Min":
            img = col.select(band).min()
            legend_title = "Min"
            if band in ["temperature_2m_max", "temperature_2m_min", "temperature_2m"]:
                img = img.subtract(273.15)
                vis = vis.copy()
                vis['min'] = vis['min'] - 273.15
                vis['max'] = vis['max'] - 273.15
        elif map_stat == "Sum":
            img = col.select(band).sum()
            legend_title = "Sum"
            # Set legend for sum based on variable type
            if band in ["precipitation", "total_precipitation_sum"]:
                vis = {'min': 0, 'max': 200, 'palette': ['#ffffff', '#00ffff', '#0080ff', '#da00ff', '#ffa400', '#ff0000']}
            elif band in ["temperature_2m_max", "temperature_2m_min", "temperature_2m"]:
                img = img.subtract(273.15)
                vis = {'min': -20, 'max': 80, 'palette': ['#000080', '#0000d9', '#4000ff', '#8000ff', '#0080ff', '#00ffff', '#00ff80', '#80ff00', '#daff00', '#ffff00', '#fff500', '#ffda00', '#ffb000', '#ffa400', '#ff4f00', '#ff2500', '#ff0a00', '#ff00ff']}
            else:
                # Generic sum legend
                vis = {'min': float(img.reduceRegion(ee.Reducer.min(), ee.Geometry(geometry_json), 1000).get(band).getInfo() or 0),
                       'max': float(img.reduceRegion(ee.Reducer.max(), ee.Geometry(geometry_json), 1000).get(band).getInfo() or 1),
                       'palette': ['#ffffff', '#00ff00', '#ff0000']}
        elif map_stat == "Trend":
            # Linear trend: fit per-pixel regression
            def add_time(image):
                date = ee.Date(image.get('system:time_start'))
                days = date.difference(ee.Date(map_start_str), 'day')
                return image.addBands(ee.Image.constant(days).rename('time'))
            col_time = col.map(add_time)
            def linear_fit(col, band):
                fit = col.select(['time', band]).reduce(ee.Reducer.linearFit())
                return fit.select('scale')
            img = linear_fit(col_time, band)
            legend_title = "Trend (per day)"
            vis = {'min': -0.1, 'max': 0.1, 'palette': ['blue', 'white', 'red']}
        elif map_stat == "Threshold":
            img = col.select(band).mean()
            if threshold_type == "Above":
                img = img.gt(threshold_val)
            else:
                img = img.lt(threshold_val)
            legend_title = f"{'>' if threshold_type=='Above' else '<'} {threshold_val}"
            vis = {'min': 0, 'max': 1, 'palette': ['white', 'red']}
        else:
            img = col.select(band).mean()
            legend_title = "Mean"
        # Clip to AOI
        img = img.clip(ee.Geometry(geometry_json))
        # Get map id and token
        map_url = img.getMapId(vis)
        # Folium map
        fmap = folium.Map(location=map_center, zoom_start=zoom_start)
        folium.raster_layers.TileLayer(
            tiles=map_url['tile_fetcher'].url_format,
            attr="Google Earth Engine",
            name=f"{era5_bands.get(band, band)} {legend_title}",
            overlay=True,
            control=True,
            opacity=0.8
        ).add_to(fmap)
        # Add AOI boundary
        try:
            if geometry_json['type'] == 'Polygon':
                coords = geometry_json['coordinates'][0]
                folium.Polygon(
                    locations=[[lat, lon] for lon, lat in coords],
                    color='red',
                    weight=2,
                    fill=False,
                    popup='Area of Interest'
                ).add_to(fmap)
        except Exception:
            pass
        # Add legend
        def add_legend(fmap, title, palette, vmin, vmax):
            import branca.colormap as cm
            # Ensure all colors start with '#'
            palette = [f"#{c}" if not c.startswith("#") else c for c in palette]
            # Ensure at least two distinct colors in palette
            if len(palette) == 1 or all(p == palette[0] for p in palette):
                palette = [palette[0], "#ffffff" if palette[0].lower() != "#ffffff" else "#000000"]
            # Avoid division by zero in colormap
            if vmin == vmax:
                vmax = vmin + 1
            colormap = cm.LinearColormap(
                colors=palette,
                vmin=vmin,
                vmax=vmax
            )
            colormap.caption = title
            fmap.add_child(colormap)
        if 'palette' in vis and 'min' in vis and 'max' in vis:
            vmin, vmax = vis['min'], vis['max']
            if vmin == vmax:
                vmax = vmin + 1e-6  # Avoid division by zero
            add_legend(fmap, f"{era5_bands.get(band, band)} {legend_title}", vis['palette'], vmin, vmax)
        st_folium(fmap, width=700, height=500)

    # Optionally, add a "Hide Map" button
    if st.button("❌ Hide Map", key="hide_map_btn"):
        st.session_state['show_map_flag'] = False
        st.experimental_rerun()

# Data Export Section
st.header("💾 Data Export")
if st.session_state.get('analysis_complete', False):
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📊 Export Statistics", type="secondary"):
            export_data = []
            for band, df in data_dict.items():
                values = df['value'].values
                if 'temperature' in band:
                    values = values - 273.15
                    unit = "°C"
                elif 'precipitation' in band and band != 'chirps_precipitation':
                    values = values * 1000
                    unit = "mm"
                elif band == 'chirps_precipitation':
                    unit = "mm"
                else:
                    unit = "units"
                stats = {
                    'Variable': era5_bands.get(band, band.replace('_', ' ').title()),
                    'Unit': unit,
                    'Count': len(values),
                    'Mean': np.mean(values),
                    'Std_Dev': np.std(values),
                    'Min': np.min(values),
                    'Max': np.max(values),
                    'Range': np.max(values) - np.min(values),
                    'Skewness': skew(values),
                    'Kurtosis': kurtosis(values),
                    'CV_Percent': np.std(values)/np.mean(values)*100
                }
                export_data.append(stats)
            export_df = pd.DataFrame(export_data)
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Statistics CSV",
                data=csv,
                file_name=f"climate_statistics_{start_date_str}_{end_date_str}.csv",
                mime="text/csv"
            )
    with col2:
        if st.button("📈 Export Time Series", type="secondary"):
            # Align all series by index using pd.concat
            series_dict = {}
            for band, df in data_dict.items():
                values = df['value'].copy()
                if 'temperature' in band:
                    values = values - 273.15
                elif 'precipitation' in band and band != 'chirps_precipitation':
                    values = values * 1000
                # Use pd.Series to preserve index
                series_dict[era5_bands.get(band, band)] = pd.Series(values, index=df.index)
            combined_df = pd.concat(series_dict, axis=1)
            csv = combined_df.to_csv()
            st.download_button(
                label="📥 Download Time Series CSV",
                data=csv,
                file_name=f"time_series_{start_date_str}_{end_date_str}.csv",
                mime="text/csv"
            )
    with col3:
        if st.button("🔗 Export Correlations", type="secondary"):
            if len(data_dict) >= 2:
                # Find common dates across all variables
                common_index = None
                for df in data_dict.values():
                    if common_index is None:
                        common_index = df.index
                    else:
                        common_index = common_index.intersection(df.index)
                if len(common_index) == 0:
                    st.error("No overlapping dates across variables for correlation analysis.")
                else:
                    correlation_data = {}
                    for band, df in data_dict.items():
                        values = df.loc[common_index, 'value'].copy()
                        if 'temperature' in band:
                            values = values - 273.15
                        elif 'precipitation' in band and band != 'chirps_precipitation':
                            values = values * 1000
                        correlation_data[era5_bands.get(band, band)] = values.values
                    corr_df = pd.DataFrame(correlation_data, index=common_index)
                    correlation_matrix = corr_df.corr()
                    csv = correlation_matrix.to_csv()
                    st.download_button(
                        label="📥 Download Correlations CSV",
                        data=csv,
                        file_name=f"climate_correlations_{start_date_str}_{end_date_str}.csv",
                        mime="text/csv"
                    )

# Footer
st.markdown("---")
st.markdown("""
**🌍 Enhanced Climate Data Dashboard (Prepared By : Dr. Anil Kumar Singh)**  
*Powered by Google Earth Engine, ERA5-Land, and CHIRPS datasets*  
*Built with Streamlit and Plotly*
""")

# Session state cleanup option
if st.sidebar.button("🔄 Reset Analysis"):
    for key in ['data_dict', 'analysis_complete', 'custom_geometry', 'drawn_geometry', 'use_drawn_geometry']:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()
