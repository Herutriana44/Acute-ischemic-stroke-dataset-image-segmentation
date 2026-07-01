"""
NUSARA Mining Inference API - Streamlit Dashboard
Interactive UI for testing and visualizing API predictions
"""

import streamlit as st
import requests
import pandas as pd
import json
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# API Configuration
API_BASE_URL = "https://herutriana44-ai-nusara-mining-api.hf.space"
API_KEY = None  # Optional API key from environment or input

# Page Configuration
st.set_page_config(
    page_title="NUSARA Mining Inference Dashboard",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF5733;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #444;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Cache for API responses
@st.cache_data(ttl=300)
def call_api(endpoint, data=None, method="GET"):
    """Call the API with optional authentication"""
    url = f"{API_BASE_URL}{endpoint}"
    headers = {"Content-Type": "application/json"}

    if API_KEY:
        headers["X-API-Key"] = API_KEY

    try:
        if method == "GET":
            response = requests.get(url, headers=headers, timeout=30)
        else:
            response = requests.post(url, headers=headers, json=data, timeout=30)

        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {e}")
        return None

def format_number(num, prefix=""):
    """Format number with thousands separator"""
    if pd.isna(num):
        return "N/A"
    return f"{prefix}{num:,.2f}"

def make_arrow_compatible(df):
    """Convert DataFrame to be Arrow-compatible by fixing mixed-type columns"""
    if df is None or df.empty:
        return df

    df_copy = df.copy()
    for col in df_copy.columns:
        if df_copy[col].dtype == 'object':
            # Convert object columns to string to avoid mixed-type issues
            df_copy[col] = df_copy[col].astype(str)

    return df_copy

def create_risk_chart(predictions):
    """Create risk level distribution chart"""
    if not predictions:
        return None

    risk_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for p in predictions:
        risk = p.get("risk_level", "LOW")
        if risk in risk_counts:
            risk_counts[risk] += 1

    colors = {"HIGH": "#FF4444", "MEDIUM": "#FFAA44", "LOW": "#44FF44"}

    fig = go.Figure(data=[
        go.Bar(
            x=list(risk_counts.keys()),
            y=list(risk_counts.values()),
            marker_color=[colors.get(k, "#888888") for k in risk_counts.keys()],
            text=list(risk_counts.values()),
            textposition="auto",
        )
    ])

    fig.update_layout(
        title="Risk Level Distribution",
        xaxis_title="Risk Level",
        yaxis_title="Count",
        height=400,
        showlegend=False
    )

    return fig

def create_priority_chart(predictions):
    """Create maintenance priority chart"""
    if not predictions:
        return None

    action_counts = {"IMMEDIATE": 0, "SCHEDULE": 0, "MONITOR": 0}
    for p in predictions:
        action = p.get("action", "MONITOR")
        if action in action_counts:
            action_counts[action] += 1

    colors = {"IMMEDIATE": "#FF0000", "SCHEDULE": "#FFA500", "MONITOR": "#00AA00"}

    fig = go.Figure(data=[
        go.Pie(
            labels=list(action_counts.keys()),
            values=list(action_counts.values()),
            marker_colors=[colors.get(k, "#888888") for k in action_counts.keys()],
            hole=0.3
        )
    ])

    fig.update_layout(
        title="Maintenance Priority Distribution",
        height=400
    )

    return fig

def create_anomaly_chart(anomalies, summary):
    """Create cost anomaly visualization"""
    if not anomalies:
        return None

    df = pd.DataFrame(anomalies)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df.get('project_name', ['N/A'] * len(df)),
        y=df.get('variance_pct', [0] * len(df)),
        marker=dict(
            color=df.get('variance_pct', [0] * len(df)),
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="Variance %")
        )
    ))

    fig.update_layout(
        title=f"Cost Anomalies (Total: {summary.get('anomalies_detected', 0)})",
        xaxis_title="Project",
        yaxis_title="Variance %",
        height=400
    )

    return fig

def create_scenario_chart(scenarios):
    """Create what-if scenario comparison chart"""
    if not scenarios:
        return None

    labels = list(scenarios.keys())
    values = list(scenarios.values())

    # Filter numeric values only
    numeric_data = []
    numeric_labels = []
    for label, value in zip(labels, values):
        if isinstance(value, (int, float)):
            numeric_labels.append(label)
            numeric_data.append(value)

    if not numeric_data:
        return None

    fig = go.Figure(data=[
        go.Bar(
            x=numeric_labels,
            y=numeric_data,
            marker_color=[['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6'][i % 5] for i in range(len(numeric_labels))],
            text=[f"{v:,.2f}" for v in numeric_data],
            textposition="auto"
        )
    ])

    fig.update_layout(
        title="What-If Scenario Results",
        xaxis_title="Scenario",
        yaxis_title="Value",
        height=400,
        xaxis_tickangle=-45
    )

    return fig

def create_fleet_cluster_chart(clusters, summary):
    """Create fleet cluster visualization"""
    if not clusters:
        return None

    df = pd.DataFrame(clusters)

    fig = px.scatter_3d(
        df,
        x='operating_hours',
        y='downtime_hours',
        z='utilization_rate',
        color='cluster',
        symbol='equipment_type',
        hover_name='equipment_name',
        hover_data=['site_name'],
        title=f"Fleet Clusters (Total: {summary.get('total_equipment', 0)})"
    )

    fig.update_layout(height=500)
    return fig

def create_fleet_priority_chart(priority_results, summary):
    """Create fleet maintenance priority chart"""
    if not priority_results:
        return None

    df = pd.DataFrame(priority_results)
    priority_counts = df['maintenance_priority'].value_counts().to_dict()

    colors = {"HIGH": "#FF4444", "MEDIUM": "#FFAA44", "LOW": "#44FF44"}

    fig = go.Figure(data=[
        go.Pie(
            labels=list(priority_counts.keys()),
            values=list(priority_counts.values()),
            marker_colors=[colors.get(k, "#888888") for k in priority_counts.keys()],
            hole=0.3
        )
    ])

    fig.update_layout(
        title=f"Fleet Maintenance Priority Distribution (Clusters: {summary.get('total_clusters', 0)})",
        height=400
    )

    return fig

def create_cluster_summary_chart(cluster_summary):
    """Create cluster characteristics summary chart"""
    if not cluster_summary:
        return None

    df = pd.DataFrame(cluster_summary)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df['cluster'],
        y=df['equipment_count'],
        name='Equipment Count'
    ))
    fig.add_trace(go.Bar(
        x=df['cluster'],
        y=df['operating_hours'],
        name='Avg Operating Hours'
    ))
    fig.add_trace(go.Bar(
        x=df['cluster'],
        y=df['downtime_hours'],
        name='Avg Downtime Hours'
    ))
    fig.add_trace(go.Bar(
        x=df['cluster'],
        y=df['utilization_rate'],
        name='Avg Utilization Rate'
    ))

    fig.update_layout(
        title="Cluster Characteristics Summary",
        xaxis_title="Cluster",
        yaxis_title="Value",
        height=400,
        barmode='group'
    )

    return fig

def display_equipment_predictions(data):
    """Display equipment failure predictions"""
    if not data or data.get("status") != "ok":
        st.warning("No equipment failure predictions available")
        return

    predictions = data.get("predictions", [])
    summary = data.get("summary", {})
    top_risk = data.get("top_risk_equipment", [])

    # Summary Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Equipment", f"{summary.get('total_records', 0):,}")
    with col2:
        st.metric("High Risk", f"{summary.get('high_risk_count', 0):,}", delta="Critical", delta_color="inverse")
    with col3:
        st.metric("Medium Risk", f"{summary.get('medium_risk_count', 0):,}", delta="Warning", delta_color="normal")
    with col4:
        st.metric("Low Risk", f"{summary.get('low_risk_count', 0):,}", delta="Safe", delta_color="normal")

    st.plotly_chart(create_risk_chart(predictions))

    # Top Risk Equipment Table
    st.subheader("🔴 Top 5 Highest Risk Equipment")
    if top_risk:
        top_df = pd.DataFrame(top_risk)
        st.dataframe(
            top_df[['equipment_name', 'site_name', 'failure_probability', 'risk_level']],
            hide_index=True
        )
    else:
        st.info("No high-risk equipment detected")

    # All Predictions (Expandable)
    with st.expander("📋 All Equipment Predictions"):
        if predictions:
            pred_df = pd.DataFrame(predictions)
            st.dataframe(
                pred_df[['equipment_name', 'site_name', 'date', 'failure_probability', 'risk_level']],
                    hide_index=True
            )

def display_maintenance_predictions(data):
    """Display maintenance priority predictions"""
    if not data or data.get("status") != "ok":
        st.warning("No maintenance priority predictions available")
        return

    predictions = data.get("predictions", [])
    maintenance_order = data.get("maintenance_order", [])

    # Summary Metrics
    immediate_count = sum(1 for p in predictions if p.get("action") == "IMMEDIATE")
    schedule_count = sum(1 for p in predictions if p.get("action") == "SCHEDULE")
    monitor_count = sum(1 for p in predictions if p.get("action") == "MONITOR")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Equipment", f"{len(predictions):,}")
    with col2:
        st.metric("Immediate Action", f"{immediate_count:,}", delta="Urgent", delta_color="inverse")
    with col3:
        st.metric("Schedule", f"{schedule_count:,}", delta="Planned", delta_color="normal")
    with col4:
        st.metric("Monitor", f"{monitor_count:,}", delta="Observe", delta_color="normal")

    st.plotly_chart(create_priority_chart(predictions))

    # Maintenance Order
    st.subheader("📅 Top 10 Maintenance Priority")
    if maintenance_order:
        order_df = pd.DataFrame(maintenance_order)
        st.dataframe(
            order_df[['equipment_name', 'site_name', 'maintenance_priority', 'action']],
            hide_index=True
        )
    else:
        st.info("No maintenance priorities available")

    # All Predictions (Expandable)
    with st.expander("📋 All Maintenance Predictions"):
        if predictions:
            pred_df = pd.DataFrame(predictions)
            st.dataframe(
                pred_df[['equipment_name', 'site_name', 'maintenance_priority', 'action']],
                    hide_index=True
            )

def display_cost_anomalies(data):
    """Display cost anomaly detection results"""
    if not data or data.get("status") != "ok":
        st.warning("No cost anomaly detection results available")
        return

    anomalies = data.get("anomalies", [])
    summary = data.get("summary", {})

    # Summary Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Transactions", f"{summary.get('total_transactions', 0):,}")
    with col2:
        st.metric("Anomalies Detected", f"{summary.get('anomalies_detected', 0):,}", delta=f"{summary.get('anomaly_rate_pct', 0)}%")
    with col3:
        st.metric("Max Variance %", f"{summary.get('max_variance_pct', 0):.2f}%")
    with col4:
        st.metric("Avg Variance %", f"{summary.get('avg_variance_pct', 0):.2f}%")

    st.plotly_chart(create_anomaly_chart(anomalies, summary))

    # Anomaly Details
    st.subheader("🚨 Detected Anomalies")
    if anomalies:
        anomaly_df = pd.DataFrame(anomalies)
        st.dataframe(
            anomaly_df[['site_name', 'project_name', 'date', 'budgeted_cost', 'actual_cost', 'variance_pct']],
            hide_index=True,
            column_config={
                "budgeted_cost": st.column_config.NumberColumn(format="Rp %d"),
                "actual_cost": st.column_config.NumberColumn(format="Rp %d"),
                "variance_pct": st.column_config.NumberColumn(format="%.2f%%")
            }
        )
    else:
        st.success("No cost anomalies detected!")

def display_whatif_simulation(data):
    """Display what-if simulation results"""
    if not data or data.get("status") != "ok":
        st.warning("No what-if simulation results available")
        return

    scenarios = data.get("scenarios", {})
    correlations = data.get("correlations", [])

    st.plotly_chart(create_scenario_chart(scenarios))

    # Scenario Details
    st.subheader("📊 Scenario Results")
    if scenarios:
        formatted_scenarios = []
        for k, v in scenarios.items():
            if isinstance(v, (int, float)):
                formatted_value = f"{v:,.2f}" if isinstance(v, float) else f"{v:,}"
            else:
                formatted_value = str(v)
            formatted_scenarios.append({"Metric": k, "Value": formatted_value})

        scenario_df = pd.DataFrame(formatted_scenarios)
        st.dataframe(scenario_df, hide_index=True)

    # Correlations
    if correlations:
        st.subheader("🔗 Strong Correlations (>0.5)")
        corr_df = pd.DataFrame(correlations)
        st.dataframe(
            corr_df[['var1', 'var2', 'correlation']],
            hide_index=True
        )

def display_fleet_optimization(data):
    """Display fleet optimization results"""
    if not data or data.get("status") != "ok":
        st.warning("No fleet optimization results available")
        return

    clusters = data.get("clusters", [])
    priority_results = data.get("maintenance_priority", [])
    cluster_summary = data.get("cluster_summary", [])
    summary = data.get("summary", {})

    # Summary Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Equipment", f"{summary.get('total_equipment', 0):,}")
    with col2:
        st.metric("Total Clusters", f"{summary.get('total_clusters', 0):,}")
    with col3:
        st.metric("High Priority", f"{summary.get('high_priority_count', 0):,}", delta="Critical", delta_color="inverse")
    with col4:
        st.metric("Medium Priority", f"{summary.get('medium_priority_count', 0):,}", delta="Warning", delta_color="normal")

    # Cluster Visualization
    st.plotly_chart(create_fleet_cluster_chart(clusters, summary))

    # Priority Distribution
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(create_fleet_priority_chart(priority_results, summary))
    with col2:
        st.plotly_chart(create_cluster_summary_chart(cluster_summary))

    # Cluster Details
    st.subheader("🔧 Fleet Cluster Details")
    if clusters:
        cluster_df = pd.DataFrame(clusters)
        st.dataframe(
            cluster_df[['equipment_name', 'site_name', 'equipment_type', 'cluster', 'operating_hours', 'downtime_hours', 'utilization_rate', 'fuel_efficiency', 'cost_per_hour']],
            hide_index=True
        )

    # Maintenance Priority Details
    st.subheader("📅 Fleet Maintenance Priority")
    if priority_results:
        priority_df = pd.DataFrame(priority_results)
        st.dataframe(
            priority_df[['equipment_name', 'site_name', 'cluster', 'maintenance_priority', 'operating_hours', 'downtime_hours', 'utilization_rate']],
            hide_index=True
        )

    # Cluster Summary
    st.subheader("📊 Cluster Summary")
    if cluster_summary:
        summary_df = pd.DataFrame(cluster_summary)
        st.dataframe(summary_df, hide_index=True)

def display_executive_summary(summary):
    """Display executive summary"""
    if not summary:
        return

    st.header("📊 Executive Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "High Risk Equipment",
            f"{summary.get('high_risk_equipment', 0):,}",
            help="Number of equipment with high failure probability"
        )
        st.metric(
            "Total Equipment Assessed",
            f"{summary.get('total_equipment_assessed', 0):,}",
            help="Total number of equipment units analyzed"
        )

    with col2:
        st.metric(
            "Urgent Maintenance Actions",
            f"{summary.get('urgent_maintenance_actions', 0):,}",
            delta="Immediate attention required",
            delta_color="inverse",
            help="Number of equipment requiring immediate maintenance"
        )
        st.metric(
            "Cost Anomalies Detected",
            f"{summary.get('cost_anomalies_detected', 0):,}",
            help="Number of financial anomalies identified"
        )

    with col3:
        baseline = summary.get('estimated_monthly_volume_baseline', 0)
        optimized = summary.get('estimated_monthly_volume_optimized', 0)

        st.metric(
            "Baseline Monthly Volume",
            f"{baseline:,.2f}",
            help="Current estimated monthly production"
        )
        st.metric(
            "Optimized Monthly Volume",
            f"{optimized:,.2f}",
            delta=f"+{((optimized - baseline) / max(baseline, 1) * 100):.1f}%" if baseline > 0 else None,
            help="Estimated production with optimizations"
        )

    # Fleet Optimization Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Fleet Clusters",
            f"{summary.get('fleet_clusters', 0):,}",
            help="Number of operational clusters identified"
        )
    with col2:
        st.metric(
            "Fleet High Priority",
            f"{summary.get('fleet_high_priority_count', 0):,}",
            delta="Critical",
            delta_color="inverse",
            help="Number of cluster segments requiring high priority maintenance"
        )

    # Recommendations
    recommendations = summary.get("recommendations", [])
    if recommendations:
        st.subheader("💡 Recommendations")
        for i, rec in enumerate(recommendations, 1):
            st.info(f"{i}. {rec}")

def custom_input_form():
    """Form for custom input data"""
    st.header("✏️ Custom Input Data")

    tab1, tab2, tab3 = st.tabs(["Equipment Data", "Financial Data", "Production Data"])

    with tab1:
        st.subheader("Equipment Features")
        num_records = st.number_input("Number of records", min_value=1, max_value=50, value=1)

        records = []
        for i in range(num_records):
            with st.expander(f"Record {i+1}"):
                cols = st.columns(4)
                with cols[0]:
                    date = st.date_input("Date", value=datetime.now(), key=f"eq_date_{i}")
                    site_name = st.text_input("Site Name", value="Tambang Nikel E", key=f"eq_site_{i}")
                with cols[1]:
                    equipment_name = st.text_input("Equipment Name", value=f"Bulldozer BD-00{i+1}", key=f"eq_name_{i}")
                    equipment_type = st.selectbox("Equipment Type", ["Bulldozer", "Excavator", "Drill", "Truck"], key=f"eq_type_{i}")
                with cols[2]:
                    operating_hours = st.number_input("Operating Hours", value=286.0, key=f"eq_ops_{i}")
                    downtime_hours = st.number_input("Downtime Hours", value=23.0, key=f"eq_down_{i}")
                with cols[3]:
                    fuel_consumption = st.number_input("Fuel Consumption", value=1667.0, key=f"eq_fuel_{i}")
                    maintenance_cost = st.number_input("Maintenance Cost", value=849403.0, key=f"eq_cost_{i}")

                records.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "site_name": site_name,
                    "equipment_name": equipment_name,
                    "equipment_type": equipment_type,
                    "operating_hours": operating_hours,
                    "downtime_hours": downtime_hours,
                    "fuel_consumption": fuel_consumption,
                    "maintenance_cost": maintenance_cost
                })

        st.session_state.custom_equipment_data = records

    with tab2:
        st.subheader("Financial Features")
        num_financial = st.number_input("Number of financial records", min_value=1, max_value=50, value=1, key="fin_count")

        financial_records = []
        for i in range(num_financial):
            with st.expander(f"Financial Record {i+1}"):
                cols = st.columns(4)
                with cols[0]:
                    date = st.date_input("Date", value=datetime.now(), key=f"fin_date_{i}")
                    site_name = st.text_input("Site Name", value="Area Tambang C", key=f"fin_site_{i}")
                with cols[1]:
                    project_name = st.text_input("Project Name", value="Project Alpha", key=f"fin_proj_{i}")
                    budgeted_cost = st.number_input("Budgeted Cost", value=171870.0, key=f"fin_budget_{i}")
                with cols[2]:
                    actual_cost = st.number_input("Actual Cost", value=127873.0, key=f"fin_actual_{i}")

                financial_records.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "site_name": site_name,
                    "project_name": project_name,
                    "budgeted_cost": budgeted_cost,
                    "actual_cost": actual_cost
                })

        st.session_state.custom_financial_data = financial_records

    with tab3:
        st.subheader("Production Features")
        num_production = st.number_input("Number of production records", min_value=1, max_value=50, value=1, key="prod_count")

        production_records = []
        for i in range(num_production):
            with st.expander(f"Production Record {i+1}"):
                cols = st.columns(3)
                with cols[0]:
                    date = st.date_input("Date", value=datetime.now(), key=f"prod_date_{i}")
                    site_name = st.text_input("Site Name", value="Area Tambang C", key=f"prod_site_{i}")
                with cols[1]:
                    material_name = st.text_input("Material Name", value="Iron", key=f"prod_mat_{i}")
                    produced_volume = st.number_input("Produced Volume", value=56.0, key=f"prod_vol_{i}")
                with cols[2]:
                    unit_cost = st.number_input("Unit Cost", value=161.0, key=f"prod_unit_{i}")

                production_records.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "site_name": site_name,
                    "material_name": material_name,
                    "produced_volume": produced_volume,
                    "unit_cost": unit_cost
                })

        st.session_state.custom_production_data = production_records

def main():
    """Main application"""

    # Header
    st.markdown('<div class="main-header">⛏️ NUSARA Mining Inference Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Interactive UI for ML-based mining operations optimization</div>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/FF5733/FFFFFF?text=NUSARA+Mining")

        st.header("⚙️ Configuration")

        # API Key Input
        global API_KEY
        api_key_input = st.text_input("API Key (Optional)", type="password", value="")
        if api_key_input:
            API_KEY = api_key_input

        # API Status Check
        st.divider()
        st.subheader("🔍 API Status")

        if st.button("Check API Health"):
            with st.spinner("Checking API..."):
                health_data = call_api("/health")
                if health_data:
                    status = health_data.get("status", "unknown")
                    models_loaded = health_data.get("model_count", 0)

                    st.success(f"API Status: {status.upper()}")
                    st.info(f"Models Loaded: {models_loaded}")

                    if models_loaded > 0:
                        st.success("✅ API is ready for predictions")
                    else:
                        st.warning("⚠️ API is running but no models loaded")
                else:
                    st.error("❌ Failed to connect to API")

        # Quick Stats
        st.divider()
        st.subheader("📊 Quick Stats")

        if st.button("Get API Info"):
            info_data = call_api("/")
            if info_data:
                st.json(info_data)

        # Navigation
        st.divider()
        st.subheader("🧭 Navigation")
        page = st.radio(
            "Select Page",
            ["Overview", "Equipment Failure", "Maintenance Priority", "Cost Anomaly", "What-If Simulation", "Fleet Optimization", "Full Pipeline", "Custom Input"]
        )

    # Main Content based on selected page
    if page == "Overview":
        st.header("📈 Dashboard Overview")

        # Check API health on page load
        with st.spinner("Loading API status..."):
            health_data = call_api("/health")

        if health_data:
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("API Status", health_data.get("status", "unknown").upper())
            with col2:
                st.metric("Models Loaded", health_data.get("model_count", 0))
            with col3:
                st.metric("Model Directory", health_data.get("model_dir", "N/A"))
            with col4:
                st.metric("Data Directory", health_data.get("data_dir", "N/A"))

        st.divider()

        st.header("🎯 Available Endpoints")

        endpoints = [
            {"name": "Equipment Failure Prediction", "path": "/predict/equipment-failure", "description": "Predict equipment failure risk (0-1 probability)"},
            {"name": "Maintenance Priority", "path": "/predict/maintenance-priority", "description": "Predict maintenance priority per equipment"},
            {"name": "Cost Anomaly Detection", "path": "/predict/cost-anomaly", "description": "Detect cost anomalies using IsolationForest"},
            {"name": "What-If Simulation", "path": "/predict/whatif-simulation", "description": "Run production what-if scenarios"},
            {"name": "Fleet Optimization", "path": "/predict/fleet-optimization", "description": "Optimize fleet operations using KMeans clustering"},
            {"name": "Full Pipeline", "path": "/predict/all", "description": "Run all models and get consolidated report"}
        ]

        endpoint_df = pd.DataFrame(endpoints)
        st.dataframe(endpoint_df, hide_index=True)

        st.divider()

        st.header("📚 API Documentation")
        st.markdown(f"""
        Access the interactive API documentation:

        - **Swagger UI**: [{API_BASE_URL}/docs]({API_BASE_URL}/docs)
        - **ReDoc**: [{API_BASE_URL}/redoc]({API_BASE_URL}/redoc)

        **Request Format:**
        ```json
        {{
          "records": [
            {{
              "date": "2025-05-01",
              "site_name": "Tambang Nikel E",
              "equipment_name": "Bulldozer BD-002",
              "equipment_type": "Bulldozer",
              "operating_hours": 286.0,
              "downtime_hours": 23.0,
              "fuel_consumption": 1667.0,
              "maintenance_cost": 849403.0
            }}
          ]
        }}
        ```

        **Note:** If `records` is omitted or empty, the API will use the bundled feature CSV files.
        """)

    elif page == "Equipment Failure":
        st.header("🔍 Equipment Failure Prediction")

        st.markdown("""
        Predict the risk of equipment failure using trained Gradient Boosting classifier.
        The model analyzes operating hours, downtime, fuel consumption, and maintenance costs
        to estimate failure probability for each equipment unit.
        """)

        if st.button("🚀 Run Equipment Failure Prediction"):
            with st.spinner("Running prediction..."):
                data = call_api("/predict/equipment-failure", data=None, method="POST")

            if data:
                display_equipment_predictions(data)
            else:
                st.error("Failed to get predictions")

    elif page == "Maintenance Priority":
        st.header("🔧 Maintenance Priority Prediction")

        st.markdown("""
        Predict maintenance priority for each equipment unit based on usage patterns,
        downtime history, and maintenance costs. Helps prioritize maintenance actions
        to prevent costly failures.
        """)

        if st.button("🚀 Run Maintenance Priority Prediction"):
            with st.spinner("Running prediction..."):
                data = call_api("/predict/maintenance-priority", data=None, method="POST")

            if data:
                display_maintenance_predictions(data)
            else:
                st.error("Failed to get predictions")

    elif page == "Cost Anomaly":
        st.header("💰 Cost Anomaly Detection")

        st.markdown("""
        Detect cost anomalies in financial transactions using Isolation Forest algorithm.
        Identifies unusual spending patterns that may indicate errors, fraud, or inefficiencies.
        """)

        if st.button("🚀 Run Cost Anomaly Detection"):
            with st.spinner("Running detection..."):
                data = call_api("/predict/cost-anomaly", data=None, method="POST")

            if data:
                display_cost_anomalies(data)
            else:
                st.error("Failed to get anomaly detection results")

    elif page == "What-If Simulation":
        st.header("🎯 What-If Simulation")

        st.markdown("""
        Run what-if scenarios to estimate the impact of operational changes on production volume.
        Simulate scenarios like increasing operating hours, reducing downtime, or combined improvements.
        """)

        if st.button("🚀 Run What-If Simulation"):
            with st.spinner("Running simulation..."):
                data = call_api("/predict/whatif-simulation", data=None, method="POST")

            if data:
                display_whatif_simulation(data)
            else:
                st.error("Failed to run simulation")

    elif page == "Full Pipeline":
        st.header("🔄 Full Pipeline Execution")

        st.markdown("""
        Run all inference models (Equipment Failure, Maintenance Priority, Cost Anomaly, What-If Simulation)
        and get a consolidated executive summary with actionable recommendations.
        """)

        if st.button("🚀 Run Full Pipeline", type="primary"):
            with st.spinner("Running full pipeline... This may take a moment..."):
                data = call_api("/predict/all", data=None, method="POST")

            if data:
                # Display executive summary first
                executive_summary = data.get("executive_summary", {})
                display_executive_summary(executive_summary)

                st.divider()

                # Display individual model results in tabs
                tab1, tab2, tab3, tab4 = st.tabs([
                    "Equipment Failure",
                    "Maintenance Priority",
                    "Cost Anomaly",
                    "What-If Simulation"
                ])

                with tab1:
                    eq_data = data.get("models", {}).get("equipment_failure", {})
                    display_equipment_predictions(eq_data)

                with tab2:
                    maint_data = data.get("models", {}).get("maintenance_priority", {})
                    display_maintenance_predictions(maint_data)

                with tab3:
                    cost_data = data.get("models", {}).get("cost_anomaly", {})
                    display_cost_anomalies(cost_data)

                with tab4:
                    whatif_data = data.get("models", {}).get("whatif_simulation", {})
                    display_whatif_simulation(whatif_data)

                # Raw JSON output
                with st.expander("🔍 Raw JSON Output"):
                    json_str = json.dumps(data, indent=2, default=str)
                    st.code(json_str, language='json')
            else:
                st.error("Failed to run full pipeline")

    elif page == "Fleet Optimization":
        st.header("🚚 Fleet Optimization")

        st.markdown("""
        Optimize fleet operations using KMeans clustering. The model analyzes operating patterns,
        downtime, fuel efficiency, and cost metrics to group similar equipment and identify
        maintenance priorities across the fleet.
        """)

        if st.button("🚀 Run Fleet Optimization"):
            with st.spinner("Running fleet optimization..."):
                data = call_api("/predict/fleet-optimization", data=None, method="POST")

            if data:
                display_fleet_optimization(data)
            else:
                st.error("Failed to get fleet optimization results")

        st.divider()
        st.markdown("""
        **Note:** Fleet optimization requires equipment data with operating hours, downtime,
        fuel consumption, and cost metrics for best results.
        """)

    elif page == "Custom Input":
        custom_input_form()

        st.divider()

        if st.button("🚀 Run Predictions with Custom Data"):
            # Prepare custom data for each endpoint
            custom_data = {}

            if "custom_equipment_data" in st.session_state:
                custom_data["equipment"] = st.session_state.custom_equipment_data

            if "custom_financial_data" in st.session_state:
                custom_data["financial"] = st.session_state.custom_financial_data

            if "custom_production_data" in st.session_state:
                custom_data["production"] = st.session_state.custom_production_data

            if not custom_data:
                st.warning("Please add custom data in the tabs above")
            else:
                with st.spinner("Running predictions with custom data..."):
                    # Run predictions for each data type
                    tabs = []

                    if "equipment" in custom_data:
                        eq_result = call_api("/predict/equipment-failure", data={"records": custom_data["equipment"]}, method="POST")
                        tabs.append(("Equipment Failure", eq_result))

                    if "equipment" in custom_data:
                        maint_result = call_api("/predict/maintenance-priority", data={"records": custom_data["equipment"]}, method="POST")
                        tabs.append(("Maintenance Priority", maint_result))

                    if "financial" in custom_data:
                        cost_result = call_api("/predict/cost-anomaly", data={"records": custom_data["financial"]}, method="POST")
                        tabs.append(("Cost Anomaly", cost_result))

                    if "production" in custom_data:
                        whatif_result = call_api("/predict/whatif-simulation", data={"records": custom_data["production"]}, method="POST")
                        tabs.append(("What-If Simulation", whatif_result))

                # Display results in tabs
                if tabs:
                    tab_names = [name for name, _ in tabs]
                    tab_results = [result for _, result in tabs]

                    result_tabs = st.tabs(tab_names)

                    for i, (tab_name, result) in enumerate(tabs):
                        with result_tabs[i]:
                            if result:
                                if "equipment_failure" in str(result.get("model", "")):
                                    display_equipment_predictions(result)
                                elif "predictive_maintenance" in str(result.get("model", "")):
                                    display_maintenance_predictions(result)
                                elif "cost_anomaly" in str(result.get("model", "")):
                                    display_cost_anomalies(result)
                                elif "whatif_simulation" in str(result.get("model", "")):
                                    display_whatif_simulation(result)
                                else:
                                    st.json(result)
                            else:
                                st.error(f"Failed to get results for {tab_name}")

    # Footer
    st.divider()
    st.markdown("""
        <div style='text-align: center; color: #888; padding: 2rem;'>
            <p>NUSARA Mining Inference API | Built with ❤️ for mining operations optimization</p>
            <p>API: <code>https://herutriana44-ai-nusara-mining-api.hf.space/</code></p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
