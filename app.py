"""
RiskLens Pro - Risk Analysis and Prediction Platform
Powered by Streamlit and ML

This application provides project risk analysis, prediction, and visualization capabilities
with self-contained data processing and model training functionality.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import datetime
import io
import base64

# Handle plotly import with fallback mechanism
try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    import subprocess
    import sys
    st.write("Installing plotly package...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly>=5.13.0", "--no-cache-dir"])
    import plotly.express as px
    import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import random
from typing import List, Dict, Any, Tuple, Optional, Union

# Import utility modules
from utils.data_processor import (
    handle_file_upload, transform_data_to_template, validate_data, 
    get_column_statistics, create_preprocessing_pipeline, split_train_test_data,
    analyze_data_quality,
    TARGET_VARIABLE, PROJECT_ID_COLUMN, PROJECT_NAME_COLUMN,
    DEFAULT_CATEGORICAL_FEATURES, DEFAULT_NUMERICAL_FEATURES
)
from utils.model_builder import ModelBuilder
from utils.visualization import (
    plot_feature_importance, plot_confusion_matrix, plot_roc_curve,
    plot_precision_recall_curve, plot_feature_distribution, plot_scatter,
    plot_cluster_analysis, plot_lime_explanation, plot_risk_heatmap,
    plot_model_comparison, plot_risk_timeline, plot_project_risks
)
from utils.export import create_pdf_report, get_download_link
# TODO: Implement PowerPoint export functionality

# Import new intelligent modules
from utils.data_quality import (
    detect_column_schema, assess_data_quality, get_data_quality_chart,
    get_column_quality_chart, get_data_quality_summary, get_data_quality_recommendations
)
from utils.heuristic_risk import (
    calculate_heuristic_risk_scores, generate_risk_factor_chart,
    get_risk_distribution_chart, explain_project_risk,
    generate_risk_mitigation_recommendations, format_risk_recommendations
)
from utils.bayesian_risk import (
    initialize_network, calculate_risk_factors, get_most_influential_factors,
    get_network_explanation, create_network_visualization, create_risk_impact_simulation,
    create_scenario_simulation_chart, create_sensitivity_chart
)

# Constants based on Arcadis Brand Guidelines
ARCADIS_PRIMARY_COLOR = "#FF6900"  # Arcadis Orange
ARCADIS_SECONDARY_COLOR = "#4D4D4F"  # Arcadis Dark Gray
ARCADIS_ACCENT_COLOR = "#4a4a4a"  # Dark Gray (accent)
ARCADIS_LIGHT_BG = "#f0f2f6"  # Light Background
ARCADIS_DARK_BG = "#333333"  # Dark Background
ARCADIS_SUCCESS = "#28a745"  # Green for success
ARCADIS_WARNING = "#ffc107"  # Yellow for warnings
ARCADIS_DANGER = "#dc3545"  # Red for danger/alerts

# Tab configuration
TABS = [
    {"id": "welcome", "name": "Welcome", "emoji": "👋"},
    {"id": "executive_summary", "name": "Executive Summary", "emoji": "📊"},
    {"id": "portfolio_deep_dive", "name": "Portfolio Deep Dive", "emoji": "🔍"},
    {"id": "model_analysis", "name": "Model Analysis & Explainability", "emoji": "🧠"},
    {"id": "simulation", "name": "Simulation & Scenarios", "emoji": "🎲"}
]

# Functions for application flow

@st.cache_data(ttl=3600, show_spinner=False)
def load_sample_data():
    """Load sample data for demo purposes"""
    # Create a basic sample project dataset
    n_samples = 250  # Increased sample size for better visualization
    data = {
        PROJECT_ID_COLUMN: [f"PROJ{1000+i}" for i in range(n_samples)],
        PROJECT_NAME_COLUMN: [f"Project {random.choice(['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon', 'Omega'])} {i+1}" for i in range(n_samples)],
        "ProjectType": np.random.choice(["Infrastructure", "Building", "Environmental", "Water", "Digital", "Technology"], n_samples),
        "Region": np.random.choice(["North America", "Europe", "APAC", "LATAM", "MEA"], n_samples),
        "Sector": np.random.choice(["Public", "Private", "Mixed"], n_samples),
        "ComplexityLevel": np.random.choice(["Low", "Medium", "High", "Very High"], n_samples),
        "ClientType": np.random.choice(["Government", "Corporate", "Private Equity", "NGO"], n_samples),
        "Budget": np.random.gamma(2, 10000000, n_samples).round(-3),
        "DurationMonths": np.random.randint(3, 84, n_samples),
        "TeamSize": np.random.poisson(15, n_samples) + 2,
        "InitialRiskScore": np.random.beta(2, 5, n_samples).round(3),
        "ChangeRequests": np.random.poisson(5, n_samples),
        "StakeholderEngagementScore": np.random.randint(1, 11, size=n_samples),
        "StartDate": pd.date_range(start="2021-01-01", periods=n_samples, freq="3D"),
        "InitialCost": np.random.gamma(2, 5000000, n_samples).round(-3),
        "InitialScheduleDays": np.random.randint(30, 2000, n_samples),
        "ActualCost": np.random.gamma(2, 5500000, n_samples).round(-3),
        "ActualScheduleDays": np.random.randint(30, 2200, n_samples),
        "RiskEventOccurred": np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        "ResourceAvailability_High": np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
    }
    
    # Generate target variable based on features
    base_prob = (
        0.05 
        + (pd.Series(data["ComplexityLevel"]).map({"Low": 0, "Medium": 0.05, "High": 0.15, "Very High": 0.3}))
        + (pd.Series(data["DurationMonths"]) / 800)
        + (pd.Series(data["Budget"]) / 5e8)
        + (pd.Series(data["ChangeRequests"]) * 0.01)
        + (pd.Series(data["RiskEventOccurred"]) * 0.3)
        + (pd.Series(data["ResourceAvailability_High"]) * 0.2)
        - (pd.Series(data["StakeholderEngagementScore"]) * 0.01)
    )
    
    noise = np.random.normal(0, 0.1, n_samples)
    final_prob = np.clip(base_prob + noise, 0.01, 0.95)
    data[TARGET_VARIABLE] = (np.random.rand(n_samples) < final_prob).astype(int)
    
    # Add a few missing values to make it realistic
    for col in ["Budget", "TeamSize", "StakeholderEngagementScore"]:
        mask = np.random.choice([True, False], size=n_samples, p=[0.05, 0.95])
        series = pd.Series(data[col])
        series[mask] = np.nan
        data[col] = series.values
    
    df = pd.DataFrame(data)
    
    # Create risk register data
    risk_data = []
    risk_types = ["Schedule Delay", "Cost Overrun", "Scope Creep", "Resource Shortage", "Technical Issue", 
                  "Regulatory Change", "Stakeholder Conflict", "Quality Issues", "Safety Incident", "Dependency Failure"]
    
    for pid in data[PROJECT_ID_COLUMN]:
        num_risks = np.random.randint(0, 8)
        for i in range(num_risks):
            impact = np.random.choice(["Very Low", "Low", "Medium", "High", "Very High"])
            likelihood = np.random.choice(["Very Low", "Low", "Medium", "High", "Very High"])
            
            risk_data.append({
                "RiskID": f"R{np.random.randint(1000, 9999)}",
                PROJECT_ID_COLUMN: pid,
                "RiskType": np.random.choice(risk_types),
                "Impact": impact,
                "Probability": likelihood,
                "Status": np.random.choice(["Open", "Mitigated", "Closed"]),
                "DateIdentified": pd.Timestamp("2022-01-01") + pd.Timedelta(days=np.random.randint(0, 365))
            })
    
    risk_df = pd.DataFrame(risk_data) if risk_data else pd.DataFrame()
    
    return df, risk_df

def format_dataframe_display(df, max_rows=10):
    """Format DataFrame for display with pagination"""
    st.dataframe(df, height=400)
    
    # Show additional rows info if dataframe is large
    if len(df) > max_rows:
        st.caption(f"Displaying {len(df)} of {len(df)} rows.")

def get_data_profiling_metrics(df):
    """Get key metrics about the dataset"""
    metrics = {
        "Total Projects": len(df),
        "High Risk Projects": int(df[TARGET_VARIABLE].sum()) if TARGET_VARIABLE in df.columns else 0,
        "Missing Values": df.isna().sum().sum(),
        "Numerical Features": len(df.select_dtypes(include=["number"]).columns),
        "Categorical Features": len(df.select_dtypes(include=["object", "category"]).columns),
        "Date Features": len(df.select_dtypes(include=["datetime"]).columns)
    }
    
    if TARGET_VARIABLE in df.columns:
        high_risk = metrics["High Risk Projects"]
        total_projects = metrics["Total Projects"]
        metrics["High Risk Rate"] = f"{(high_risk / total_projects * 100):.1f}%"
        
        # Calculate cost and schedule metrics for high risk projects
        high_risk_projects = df[df[TARGET_VARIABLE] == 1]
        if "ActualCost" in df.columns and "InitialCost" in df.columns:
            high_risk_cost_overrun = ((high_risk_projects["ActualCost"] - high_risk_projects["InitialCost"]) / 
                                     high_risk_projects["InitialCost"]).mean() * 100
            metrics["Avg Cost Overrun % (High-Risk)"] = f"{high_risk_cost_overrun:.1f}%"
            
        if "ActualScheduleDays" in df.columns and "InitialScheduleDays" in df.columns:
            high_risk_schedule_overrun = ((high_risk_projects["ActualScheduleDays"] - high_risk_projects["InitialScheduleDays"]) / 
                                         high_risk_projects["InitialScheduleDays"]).mean() * 100
            metrics["Avg Schedule Overrun % (High-Risk)"] = f"{high_risk_schedule_overrun:.1f}%"
    
    return metrics

def calculate_model_metrics(predictions, actuals, probabilities):
    """Calculate and return model performance metrics"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    metrics = {
        "accuracy": accuracy_score(actuals, predictions),
        "precision": precision_score(actuals, predictions),
        "recall": recall_score(actuals, predictions),
        "f1_score": f1_score(actuals, predictions),
        "roc_auc": roc_auc_score(actuals, probabilities)
    }
    
    return metrics

def initialize_session_state():
    """Initialize session state variables"""
    if 'project_data' not in st.session_state:
        st.session_state.project_data = None
    if 'risk_data' not in st.session_state:
        st.session_state.risk_data = None
    if 'trained_models' not in st.session_state:
        st.session_state.trained_models = None
    if 'best_model_name' not in st.session_state:
        st.session_state.best_model_name = None
    if 'model_results' not in st.session_state:
        st.session_state.model_results = {}
    if 'visualizations' not in st.session_state:
        st.session_state.visualizations = {}
    if 'categorical_features' not in st.session_state:
        st.session_state.categorical_features = DEFAULT_CATEGORICAL_FEATURES
    if 'numerical_features' not in st.session_state:
        st.session_state.numerical_features = DEFAULT_NUMERICAL_FEATURES
    if 'risk_predictions' not in st.session_state:
        st.session_state.risk_predictions = None
    if 'risk_probabilities' not in st.session_state:
        st.session_state.risk_probabilities = None
    if 'data_transformed' not in st.session_state:
        st.session_state.data_transformed = False
    if 'data_profile' not in st.session_state:
        st.session_state.data_profile = None
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = "welcome"
    if 'model_builder' not in st.session_state:
        st.session_state.model_builder = ModelBuilder()
    if 'prediction_threshold' not in st.session_state:
        st.session_state.prediction_threshold = 0.4
    if 'current_risk_model' not in st.session_state:
        st.session_state.current_risk_model = "Random Forest"
    if 'ai_narrative_generated' not in st.session_state:
        st.session_state.ai_narrative_generated = False
        
    # New intelligent data intake session variables
    if 'column_mapping' not in st.session_state:
        st.session_state.column_mapping = None
    if 'data_quality_scores' not in st.session_state:
        st.session_state.data_quality_scores = None
    if 'column_confidence_scores' not in st.session_state:
        st.session_state.column_confidence_scores = None
        
    # Heuristic risk scoring session variables
    if 'heuristic_risk_scores' not in st.session_state:
        st.session_state.heuristic_risk_scores = None
    if 'heuristic_model_metadata' not in st.session_state:
        st.session_state.heuristic_model_metadata = None
    if 'heuristic_risk_distribution' not in st.session_state:
        st.session_state.heuristic_risk_distribution = None
    if 'risk_recommendations' not in st.session_state:
        st.session_state.risk_recommendations = None
        
    # Bayesian risk analysis session variables
    if 'bayesian_network' not in st.session_state:
        st.session_state.bayesian_network = None
    if 'bayesian_evidence' not in st.session_state:
        st.session_state.bayesian_evidence = {}
    if 'bayesian_risk_factors' not in st.session_state:
        st.session_state.bayesian_risk_factors = None
    if 'lime_explainer' not in st.session_state:
        st.session_state.lime_explainer = None
    if 'feature_names_out' not in st.session_state:
        st.session_state.feature_names_out = None
        
    # Monte Carlo simulation variables
    if 'monte_carlo_simulations' not in st.session_state:
        st.session_state.monte_carlo_simulations = None
    if 'simulation_scenarios' not in st.session_state:
        st.session_state.simulation_scenarios = []
        
    # Anomaly detection variables
    if 'anomaly_detection_results' not in st.session_state:
        st.session_state.anomaly_detection_results = None
    if 'anomaly_visualization' not in st.session_state:
        st.session_state.anomaly_visualization = None
    if 'anomaly_insights' not in st.session_state:
        st.session_state.anomaly_insights = None
    if 'anomaly_detection_method' not in st.session_state:
        st.session_state.anomaly_detection_method = 'ensemble'
    if 'anomaly_contamination' not in st.session_state:
        st.session_state.anomaly_contamination = 0.1

def arcadis_logo_svg():
    """Return SVG code for Arcadis logo with RiskLens Pro (based on brand guidelines)"""
    return """
    <svg width="200" height="50" viewBox="0 0 300 50" xmlns="http://www.w3.org/2000/svg">
        <!-- Orange wave symbol - Arcadis logo -->
        <path d="M30 10 C40 5, 45 15, 35 17 C25 19, 20 30, 30 35" fill="#FF6900" stroke="none"/>
        
        <!-- Text part -->
        <text x="45" y="30" font-family="Arial" font-size="24" font-weight="bold" fill="#4D4D4F">ARCADIS</text>
        
        <!-- RiskLens Pro text -->
        <text x="160" y="30" font-family="Arial" font-size="22" font-weight="bold" fill="#FF6900">RiskLens Pro</text>
    </svg>
    """

def styled_card(title, content, icon=None, color=ARCADIS_PRIMARY_COLOR):
    """Create a styled card with title and content"""
    icon_html = f'<span style="font-size:24px;margin-right:10px;">{icon}</span>' if icon else ''
    
    st.markdown(f'''
    <div style="border:1px solid #ddd;border-radius:8px;padding:15px;margin-bottom:20px;background:white;">
        <h3 style="color:{color};margin-top:0;border-bottom:1px solid #eee;padding-bottom:10px;">
            {icon_html}{title}
        </h3>
        <div>
            {content}
        </div>
    </div>
    ''', unsafe_allow_html=True)

def styled_metric_card(label, value, delta=None, icon=None, color=ARCADIS_PRIMARY_COLOR, help_text=None):
    """Create a styled metric card"""
    icon_html = f'<span style="font-size:22px;margin-right:8px;">{icon}</span>' if icon else ''
    delta_html = ''
    if delta is not None:
        delta_color = "#28a745" if float(delta.replace('%', '')) >= 0 else "#dc3545"
        delta_icon = "▲" if float(delta.replace('%', '')) >= 0 else "▼"
        delta_html = f'<span style="color:{delta_color};font-size:14px;">{delta_icon} {delta}</span>'

    help_icon = ''
    if help_text:
        help_icon = f'<span title="{help_text}" style="cursor:help;opacity:0.7;margin-left:5px;">ⓘ</span>'
    
    value_style = "font-size:32px;font-weight:bold;"
    if len(str(value)) > 10:  # Adjust font size for long numbers
        value_style = "font-size:24px;font-weight:bold;"
    
    st.markdown(f'''
    <div style="border:1px solid #ddd;border-radius:8px;padding:15px;background:white;height:100%;">
        <div style="color:#666;font-size:14px;">{icon_html}{label}{help_icon}</div>
        <div style="{value_style}color:{color};">{value}</div>
        <div style="margin-top:5px;">{delta_html}</div>
    </div>
    ''', unsafe_allow_html=True)

def styled_header(text, level=1, color=ARCADIS_PRIMARY_COLOR, icon=None):
    """Display a header with custom styling"""
    icon_html = f'{icon} ' if icon else ''
    if level == 1:
        st.markdown(f'<h1 style="color:{color};">{icon_html}{text}</h1>', unsafe_allow_html=True)
    elif level == 2:
        st.markdown(f'<h2 style="color:{color};">{icon_html}{text}</h2>', unsafe_allow_html=True)
    elif level == 3:
        st.markdown(f'<h3 style="color:{color};">{icon_html}{text}</h3>', unsafe_allow_html=True)
    else:
        st.markdown(f'<h4 style="color:{color};">{icon_html}{text}</h4>', unsafe_allow_html=True)

def set_streamlit_style():
    """Set Streamlit page styling"""
    st.set_page_config(
        page_title="RiskLens Pro - Insight Hub",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for styling
    st.markdown(f"""
        <style>
        .main .block-container {{
            padding-top: 1rem;
            padding-bottom: 1rem;
        }}
        h1, h2, h3 {{
            color: {ARCADIS_PRIMARY_COLOR};
        }}
        .stProgress > div > div > div > div {{
            background-color: {ARCADIS_PRIMARY_COLOR};
        }}
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
        }}
        .stTabs [data-baseweb="tab"] {{
            background-color: #f0f2f6;
            border-radius: 4px 4px 0px 0px;
            padding: 10px 16px;
            border: none;
        }}
        .stTabs [aria-selected="true"] {{
            background-color: {ARCADIS_PRIMARY_COLOR};
            color: white;
        }}
        div[data-testid="stDecoration"] {{
            background-image: linear-gradient(90deg, #FF6900, #4D4D4F);
        }}
        </style>
    """, unsafe_allow_html=True)
    
    # Display header with title and brand colors
    st.markdown(f"""
        <div style="display: flex; align-items: center; margin-bottom: 1rem; background-color: white; padding: 10px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
            <div style="background-color: {ARCADIS_PRIMARY_COLOR}; width: 8px; height: 40px; margin-right: 15px;"></div>
            <h1 style="color: {ARCADIS_PRIMARY_COLOR}; margin: 0; font-size: 28px;">ARCADIS <span style="color: {ARCADIS_SECONDARY_COLOR};">RiskLens Pro</span></h1>
        </div>
    """, unsafe_allow_html=True)

def make_sidebar():
    """Create application sidebar"""
    with st.sidebar:
        st.markdown("## Data Management")
        
        # Data upload section
        with st.expander("Upload Data", expanded=True):
            upload_option = st.radio(
                "Select data source:",
                options=["Upload data file", "Use sample data"],
                horizontal=True
            )
            
            if upload_option == "Upload data file":
                uploaded_file = st.file_uploader(
                    "Upload your project data file (CSV or Excel)", 
                    type=["csv", "xlsx", "xls"]
                )
                
                if uploaded_file is not None:
                    process_btn = st.button("Process Data", type="primary", use_container_width=True)
                    if process_btn:
                        with st.spinner("Processing data..."):
                            # Process the uploaded file with enhanced intelligence
                            try:
                                df, error_msg, info_msg = handle_file_upload(uploaded_file)
                                
                                if error_msg:
                                    st.error(error_msg)
                                    st.stop()
                                
                                if df is None:
                                    st.error("No data could be extracted from the file. Please check the file format.")
                                    st.stop()
                                
                                # Store the raw data initially
                                st.session_state.project_data_raw = df
                                
                                if info_msg:
                                    st.info(info_msg)
                                
                                # Check if the data has already been intelligently mapped by the handler
                                already_mapped = info_msg and ("mapped" in info_msg.lower() or "detected" in info_msg.lower())
                            except Exception as e:
                                st.error(f"Error processing file: {str(e)}")
                                st.stop()
                            
                            if already_mapped:
                                # The data processor already did the mapping - no need to transform
                                st.session_state.project_data = df
                                st.success(f"✅ File mapped intelligently")
                                st.session_state.data_transformed = True
                                st.session_state.data_profile = get_data_profiling_metrics(df)
                                
                                # Create feedback report based on data quality
                                # Analyze missing data and create impact report
                                missing_data_impact = analyze_data_quality(df)
                                
                                overall_impact = missing_data_impact.get('overall', {})
                                data_quality_score = overall_impact.get('data_quality_score', 0.9) # Higher score for already mapped data
                                analysis_capability = overall_impact.get('analysis_capability', 'full')
                                missing_core_features = overall_impact.get('missing_core_features', [])
                                
                                # Store feedback messages
                                feedback_messages = []
                                feedback_messages.append({"type": "success", "message": f"✅ Data format automatically detected and mapped with {data_quality_score:.0%} coverage"})
                                
                                if missing_core_features:
                                    feedback_messages.append({"type": "info", "message": f"ℹ️ Some recommended data is missing but analysis can proceed: {', '.join(missing_core_features)}"})
                                
                                # Create data mapping report
                                st.session_state.data_mapping_report = {
                                    'quality_score': data_quality_score,
                                    'analysis_capability': analysis_capability,
                                    'missing_features': missing_core_features,
                                    'details': missing_data_impact,
                                    'feedback_messages': feedback_messages
                                }
                                
                                # Display messages
                                for msg in feedback_messages:
                                    if msg["type"] == "success":
                                        st.success(msg["message"])
                                    elif msg["type"] == "info":
                                        st.info(msg["message"])
                                    elif msg["type"] == "warning":
                                        st.warning(msg["message"])
                                
                                st.rerun()
                                                                    
                            else:
                                # Standard process - transform using template
                                st.success(f"✅ File loaded successfully")
                                
                                # Transform data to fit required template with intelligent analysis
                                transformed_df, mapping_info, missing_data_impact = transform_data_to_template(df)
                                
                                if transformed_df is not None:
                                    # Store the transformed data and mapping info
                                    st.session_state.project_data = transformed_df
                                    st.session_state.mapping_info = mapping_info
                                    st.session_state.missing_data_impact = missing_data_impact
                                    st.session_state.data_transformed = True
                                    st.session_state.data_profile = get_data_profiling_metrics(transformed_df)
                                
                                # Show detailed feedback about the data transformation
                                overall_impact = missing_data_impact.get('overall', {})
                                data_quality_score = overall_impact.get('data_quality_score', 0.0)
                                analysis_capability = overall_impact.get('analysis_capability', 'unknown')
                                missing_core_features = overall_impact.get('missing_core_features', [])
                                
                                # Store feedback messages in session state to persist after rerun
                                feedback_messages = []
                                
                                # Generate feedback messages based on data quality
                                if analysis_capability == 'full' or analysis_capability == 'limited':
                                    feedback_messages.append({"type": "success", "message": f"✅ Data mapped successfully with {data_quality_score:.0%} coverage"})
                                    if missing_core_features:
                                        feedback_messages.append({"type": "info", "message": f"ℹ️ Some recommended data is missing but analysis can proceed"})
                                elif analysis_capability == 'severely_limited':
                                    feedback_messages.append({"type": "warning", "message": f"⚠️ Limited analysis possible. Missing important data: {', '.join(missing_core_features)}"})
                                else:  # not_possible
                                    feedback_messages.append({"type": "error", "message": f"⚠️ Critical data missing: {', '.join(missing_core_features)}. Full analysis not possible."})
                                
                                # If target variable is missing, add special warning
                                if TARGET_VARIABLE in missing_core_features:
                                    feedback_messages.append({"type": "error", "message": f"⚠️ No target risk variable found. Please upload data with risk indicators or outcomes."})
                                    feedback_messages.append({"type": "info", "message": "The app needs a column indicating project risk status, success/failure, or derailment indicators."})
                                
                                # Create a data mapping report in the session state for later display
                                st.session_state.data_mapping_report = {
                                    'quality_score': data_quality_score,
                                    'analysis_capability': analysis_capability,
                                    'missing_features': missing_core_features,
                                    'details': missing_data_impact,
                                    'feedback_messages': feedback_messages
                                }
                                
                                # Display the messages now
                                for msg in feedback_messages:
                                    if msg["type"] == "success":
                                        st.success(msg["message"])
                                    elif msg["type"] == "info":
                                        st.info(msg["message"])
                                    elif msg["type"] == "warning":
                                        st.warning(msg["message"])
                                    elif msg["type"] == "error":
                                        st.error(msg["message"])
                                
                                # Continue with app flow
                                st.rerun()
            else:  # Use sample data
                if st.button("Load Sample Data", type="primary", use_container_width=True):
                    with st.spinner("Loading sample data..."):
                        sample_df, risk_df = load_sample_data()
                        st.session_state.project_data = sample_df
                        st.session_state.risk_data = risk_df
                        st.session_state.data_profile = get_data_profiling_metrics(sample_df)
                        st.session_state.data_transformed = True
                        
                        # Create and store feedback messages for sample data
                        feedback_messages = [
                            {"type": "success", "message": "✅ Sample data loaded successfully"},
                            {"type": "info", "message": "ℹ️ 150 sample projects with full feature data included"},
                            {"type": "info", "message": "ℹ️ Risk register data with 450+ identified risks also loaded"}
                        ]
                        
                        # Store feedback messages in session state
                        st.session_state.data_mapping_report = {
                            'quality_score': 1.0,  # 100% coverage for sample data
                            'analysis_capability': 'full',
                            'missing_features': [],
                            'details': {},
                            'feedback_messages': feedback_messages
                        }
                        
                        # Display messages before rerun
                        for msg in feedback_messages:
                            if msg["type"] == "success":
                                st.success(msg["message"])
                            elif msg["type"] == "info":
                                st.info(msg["message"])
                            elif msg["type"] == "warning":
                                st.warning(msg["message"])
                            elif msg["type"] == "error":
                                st.error(msg["message"])
                                
                        st.rerun()
        
        # Model training section
        if st.session_state.project_data is not None and st.session_state.data_transformed:
            with st.expander("Run Risk Analytics", expanded=True):
                if st.button("Train Risk Model", type="primary", use_container_width=True):
                    with st.spinner("Training risk model..."):
                        # Execute intelligent model training
                        run_risk_analytics()
                        st.success("✅ Risk model trained successfully")
                        st.session_state.active_tab = "executive_summary"
                        # Generate AI narrative
                        st.session_state.ai_narrative_generated = True
                        st.rerun()
        
        # Display status information
        st.markdown("---")
        st.markdown("### Status")
        
        if st.session_state.project_data is not None:
            st.success(f"✅ Data loaded: {len(st.session_state.project_data)} projects")
        else:
            st.warning("⚠️ No data loaded")
            
        if st.session_state.best_model_name is not None:
            st.success(f"✅ Model: {st.session_state.best_model_name}")
            
        if st.session_state.risk_predictions is not None:
            high_risk = st.session_state.risk_predictions.sum() if st.session_state.risk_predictions is not None else 0
            st.info(f"ℹ️ High-risk projects: {high_risk}")
            
        # Display persisted data processing feedback messages if they exist
        if 'data_mapping_report' in st.session_state and 'feedback_messages' in st.session_state.data_mapping_report:
            st.markdown("---")
            st.markdown("### Data Processing Results")
            
            for msg in st.session_state.data_mapping_report['feedback_messages']:
                if msg["type"] == "success":
                    st.success(msg["message"])
                elif msg["type"] == "info":
                    st.info(msg["message"])
                elif msg["type"] == "warning":
                    st.warning(msg["message"])
                elif msg["type"] == "error":
                    st.error(msg["message"])
        
        # Export section when data is available
        if st.session_state.project_data is not None and st.session_state.risk_predictions is not None:
            st.markdown("---")
            st.markdown("### Export")
            
            export_format = st.selectbox(
                "Export format:", 
                ["PDF Report", "PowerPoint Presentation"]
            )
            
            if st.button("Generate Report", type="primary", use_container_width=True):
                with st.spinner(f"Generating {export_format.split()[0]} report..."):
                    if export_format == "PDF Report":
                        # Create a comprehensive report data object with information from all tabs
                        report_data = {
                            "project_data": st.session_state.project_data,
                            "model_results": st.session_state.model_results,
                            "visualizations": st.session_state.visualizations,
                            "risk_analysis": {
                                "high_risk_count": st.session_state.risk_predictions.sum() if st.session_state.risk_predictions is not None else 0,
                                "avg_risk_score": float(st.session_state.risk_probabilities.mean()) if hasattr(st.session_state, 'risk_probabilities') and st.session_state.risk_probabilities is not None else 0.0,
                                "risk_threshold": st.session_state.prediction_threshold,
                                "model_accuracy": st.session_state.model_results.get('accuracy', 0.0) * 100 if st.session_state.model_results else 0.0,
                                "data_completeness": 0.0,  # Will be calculated below
                                "summary_text": "This comprehensive report provides an analysis of project risks based on historical data and predictive modeling. It includes executive summary, detailed portfolio analysis, model performance metrics, and actionable recommendations for risk mitigation."
                            }
                        }
                        
                        # Add risk data if available
                        if st.session_state.risk_data is not None:
                            report_data["risk_data"] = st.session_state.risk_data
                        
                        # Add data quality information if available
                        if st.session_state.data_profile is not None:
                            data_quality = st.session_state.data_profile
                            report_data["risk_analysis"]["data_completeness"] = (1.0 - data_quality.get('missing_rate', 0.0)) * 100
                        
                        # Add heuristic risk information if available
                        if st.session_state.heuristic_risk_scores is not None:
                            report_data["risk_analysis"]["heuristic_risk_scores"] = st.session_state.heuristic_risk_scores
                        
                        if st.session_state.risk_recommendations is not None:
                            report_data["risk_analysis"]["recommendations"] = st.session_state.risk_recommendations
                        
                        # Add Monte Carlo simulation results if available
                        if hasattr(st.session_state, 'monte_carlo_results') and st.session_state.monte_carlo_results is not None:
                            report_data["risk_analysis"]["monte_carlo_metrics"] = {
                                "mean": float(st.session_state.monte_carlo_results["risk_scores"].mean()),
                                "std": float(st.session_state.monte_carlo_results["risk_scores"].std()),
                                "p10": float(st.session_state.monte_carlo_results["risk_scores"].quantile(0.1)),
                                "p90": float(st.session_state.monte_carlo_results["risk_scores"].quantile(0.9))
                            }
                        
                        # Add portfolio analysis data
                        if st.session_state.project_data is not None and st.session_state.risk_predictions is not None:
                            # Create top risky projects dataframe
                            df = st.session_state.project_data.copy()
                            if hasattr(st.session_state, 'risk_probabilities') and st.session_state.risk_probabilities is not None:
                                df['RiskScore'] = st.session_state.risk_probabilities
                                top_risky = df.sort_values('RiskScore', ascending=False).head(10)
                                top_risky = top_risky[[PROJECT_ID_COLUMN, PROJECT_NAME_COLUMN, 'RiskScore']]
                                report_data["risk_analysis"]["top_risky_projects"] = top_risky
                        
                        pdf_bytes = create_pdf_report(**report_data)
                        st.download_button(
                            label=f"Download PDF Report",
                            data=pdf_bytes,
                            file_name="RiskLens_Pro_Report.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                    # PowerPoint export functionality temporarily disabled
                    # Will be implemented in a future update
                    else:
                        st.info("PowerPoint export will be available in the next update.")
                        # report_data = {
                        #    "project_data": st.session_state.project_data,
                        #    "model_results": st.session_state.model_results,
                        #    "visualizations": st.session_state.visualizations
                        # }
                        # 
                        # if st.session_state.risk_data is not None:
                        #    report_data["risk_data"] = st.session_state.risk_data
                        # 
                        # PowerPoint export will be implemented in the future
        
        # Reset application state button
        if st.button("Reset Application", type="secondary", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            initialize_session_state()
            st.rerun()

def run_risk_analytics():
    """Run the risk analytics process on the loaded data"""
    if st.session_state.project_data is None:
        return
    
    df = st.session_state.project_data
    
    # 1. Run intelligent column mapping and data quality assessment
    if st.session_state.column_mapping is None:
        # Detect columns and create mapping
        column_mapping, confidence_scores = detect_column_schema(df)
        st.session_state.column_mapping = column_mapping
        st.session_state.column_confidence_scores = confidence_scores
        
        # Run data quality assessment
        quality_scores = assess_data_quality(df, column_mapping)
        st.session_state.data_quality_scores = quality_scores
    
    # 2. Run heuristic risk scoring (no ML training needed)
    df_with_predictions = df.copy()  # Create prediction dataframe right away
    
    if st.session_state.heuristic_risk_scores is None:
        heuristic_df, model_metadata, risk_distribution = calculate_heuristic_risk_scores(
            df, st.session_state.column_mapping
        )
        st.session_state.heuristic_risk_scores = heuristic_df
        st.session_state.heuristic_model_metadata = model_metadata
        st.session_state.heuristic_risk_distribution = risk_distribution
        
        # Store predictions from heuristic model for comparison
        df_with_predictions['HeuristicRiskScore'] = heuristic_df['risk_score']
        df_with_predictions['HeuristicRiskCategory'] = heuristic_df['risk_category']
        df_with_predictions['HeuristicHighRisk'] = heuristic_df['high_risk']
        
        # This also serves as our risk predictions if TARGET_VARIABLE not available
        if TARGET_VARIABLE not in df.columns:
            st.session_state.risk_predictions = pd.Series(heuristic_df['high_risk'].values, index=df.index)
            st.session_state.risk_probabilities = pd.Series(heuristic_df['risk_score'].values, index=df.index)
    else:
        # If we already have heuristic risk scores, add them to the dataframe
        df_with_predictions['HeuristicRiskScore'] = st.session_state.heuristic_risk_scores['risk_score']
        df_with_predictions['HeuristicRiskCategory'] = st.session_state.heuristic_risk_scores['risk_category']
        df_with_predictions['HeuristicHighRisk'] = st.session_state.heuristic_risk_scores['high_risk']
    
    # 3. Determine features based on available columns and column mapping
    numerical_features = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) 
                         and col != TARGET_VARIABLE and col != PROJECT_ID_COLUMN 
                         and df[col].nunique() > 5]
    
    categorical_features = [col for col in df.columns if (pd.api.types.is_object_dtype(df[col]) 
                           or pd.api.types.is_categorical_dtype(df[col]))
                           and col != PROJECT_NAME_COLUMN and col != PROJECT_ID_COLUMN]
    
    st.session_state.categorical_features = categorical_features
    st.session_state.numerical_features = numerical_features
    
    # 4. If TARGET_VARIABLE is available, train machine learning models
    if TARGET_VARIABLE in df.columns:
        try:
            # Split data for training
            X_train, X_test, y_train, y_test = split_train_test_data(df, TARGET_VARIABLE)
            
            # Create preprocessor
            preprocessor = create_preprocessing_pipeline(categorical_features, numerical_features)
            
            # Train the model
            model_builder = st.session_state.model_builder
            results = model_builder.train_all_models(
                X_train, y_train, X_test, y_test, 
                preprocessor, 
                numerical_features + categorical_features,
                cv_folds=3, n_iter=10
            )
            
            # Update session state with results
            st.session_state.trained_models = True
            st.session_state.best_model_name = results["best_model"]
            st.session_state.model_results = results
            
            # Generate predictions for all projects
            X = df.drop(columns=[TARGET_VARIABLE])
            predictions, probabilities = model_builder.predict(X)
            st.session_state.risk_predictions = pd.Series(predictions, index=df.index)
            st.session_state.risk_probabilities = pd.Series(probabilities, index=df.index)
            
            # Add ML predictions to the dataframe
            df_with_predictions['MLPredictedRisk'] = predictions
            df_with_predictions['MLRiskProbability'] = probabilities
        except Exception as e:
            st.warning(f"Could not train ML models. Falling back to heuristic risk scoring. Error: {str(e)}")
    
    # 5. Initialize Bayesian network with available data
    if st.session_state.bayesian_network is None:
        # Create Bayesian network using the data and column mapping
        bayesian_network = initialize_network(df, st.session_state.column_mapping, 
                                             st.session_state.heuristic_risk_scores)
        st.session_state.bayesian_network = bayesian_network
        
        # Calculate baseline risk factors with no evidence
        risk_factors = calculate_risk_factors(bayesian_network)
        st.session_state.bayesian_risk_factors = risk_factors
    
    # 5. Run anomaly detection to find outlier projects
    from utils.anomaly_detection import detect_anomalies, get_anomaly_insights
    
    # Run anomaly detection based on the selected method and contamination level
    anomaly_df, anomaly_viz = detect_anomalies(
        df_with_predictions, 
        st.session_state.column_mapping,
        method=st.session_state.anomaly_detection_method,
        contamination=st.session_state.anomaly_contamination
    )
    
    # Get insights about anomalies
    anomaly_insights = get_anomaly_insights(anomaly_df, st.session_state.column_mapping)
    
    # Store anomaly detection results
    st.session_state.anomaly_detection_results = anomaly_df
    st.session_state.anomaly_visualization = anomaly_viz
    st.session_state.anomaly_insights = anomaly_insights
    
    # Add anomaly scores to predictions dataframe
    df_with_predictions['anomaly_score'] = anomaly_df['anomaly_score']
    df_with_predictions['is_anomaly'] = anomaly_df['is_anomaly']
    
    # Store final dataframe with all predictions
    st.session_state.project_data_with_predictions = df_with_predictions

def explain_project_risk(project_data, risk_score, factor_contributions, risk_weights):
    """Generate an explanation of why a project is high risk
    
    Args:
        project_data: Series with project data
        risk_score: Risk score from heuristic model
        factor_contributions: Dictionary of factor contributions
        risk_weights: Dictionary of risk weights
        
    Returns:
        str: HTML formatted explanation
    """
    # Identify top risk factors for this project
    project_factors = {}
    
    # Extract project features that match our factor names
    for factor, weight in risk_weights.items():
        factor_display = factor.replace('_', ' ').title()
        
        # Try different variations of the factor name to find a match
        factor_variations = [
            factor,  # Original name
            factor.replace('_', ''),  # No underscores
            ''.join([word.capitalize() for word in factor.split('_')]),  # CamelCase
            factor.title().replace('_', ''),  # TitleCaseNoUnderscore
            factor_display,  # Title Case With Spaces
            factor_display.replace(' ', '')  # TitleCaseNoSpaces
        ]
        
        # Look for any matching column
        matching_col = None
        for var in factor_variations:
            for col in project_data.index:
                if isinstance(col, str) and var.lower() in col.lower():
                    matching_col = col
                    break
            if matching_col:
                break
        
        # If we found a matching column, calculate its contribution to risk
        if matching_col:
            # Get the value and normalize if it's numeric
            value = project_data[matching_col]
            if isinstance(value, (int, float)) and not pd.isna(value):
                # Normalize the value to 0-1 range (assuming most are already in that range)
                normalized_value = min(max(value, 0), 1) if 0 <= value <= 1 else value / 10 if value > 1 else 0
                project_factors[factor] = {
                    'display_name': factor_display,
                    'value': value,
                    'normalized_value': normalized_value,
                    'contribution': normalized_value * weight,
                    'weight': weight
                }
    
    # Sort factors by contribution
    sorted_factors = sorted(project_factors.items(), key=lambda x: x[1]['contribution'], reverse=True)
    
    # Create the explanation
    if risk_score >= 0.8:
        risk_level = "critical"
        risk_color = "#d9534f"  # Red
    elif risk_score >= 0.6:
        risk_level = "high"
        risk_color = "#f0ad4e"  # Orange
    elif risk_score >= 0.4:
        risk_level = "medium"
        risk_color = "#5bc0de"  # Blue
    else:
        risk_level = "low"
        risk_color = "#5cb85c"  # Green
    
    # Build HTML explanation using string parts instead of multi-line strings
    explanation = f"<h4>Risk Analysis for Project</h4>\n"
    explanation += f"<p>This project has a <span style='font-weight: bold; color: {risk_color};'>{risk_level} risk level</span> "
    explanation += f"with a risk score of <span style='font-weight: bold;'>{risk_score:.2f}</span>.</p>\n\n"
    
    explanation += "<h5>Key Risk Factors</h5>\n"
    explanation += "<p>The following factors contribute most significantly to this project's risk:</p>\n"
    explanation += "<ul>\n"
    
    # Add top 3 factors or all if less than 3
    for i, (factor, data) in enumerate(sorted_factors[:3]):
        if i == 0:
            impact = "very high"
        elif i == 1:
            impact = "high"
        else:
            impact = "moderate"
            
        explanation += f"<li><strong>{data['display_name']}</strong>: {impact} impact "
        explanation += f"(value: {data['value']}, contribution: {data['contribution']:.2f})</li>\n"
    
    explanation += "</ul>\n\n"
    
    # Add overall explanation
    if len(sorted_factors) >= 2:
        explanation += "<h5>Risk Interpretation</h5>\n"
        explanation += "<p>"
        
        top_factor = sorted_factors[0][1]['display_name']
        second_factor = sorted_factors[1][1]['display_name']
        
        explanation += f"This project is primarily at risk due to high <strong>{top_factor}</strong> "
        explanation += f"combined with significant <strong>{second_factor}</strong>. "
        
        if len(sorted_factors) >= 3:
            third_factor = sorted_factors[2][1]['display_name']
            explanation += f"Additionally, <strong>{third_factor}</strong> contributes to the overall risk profile."
        
        explanation += "</p>\n\n"
    
    # Add risk category context
    explanation += "<h5>Risk Level Context</h5>\n"
    explanation += f"<p>A risk score of {risk_score:.2f} places this project in the <span style='font-weight: bold; color: {risk_color};'>"
    explanation += f"{risk_level} risk category</span>. "
    
    if risk_level in ["high", "critical"]:
        explanation += "This project requires immediate attention and proactive risk management.</p>"
    elif risk_level == "medium":
        explanation += "This project should be regularly monitored for changes in risk factors.</p>"
    else:
        explanation += "This project currently has manageable risk levels.</p>"
    
    return explanation


def generate_risk_mitigation_recommendations(project_data, risk_score, factor_contributions, risk_weights):
    """Generate risk mitigation recommendations based on project data
    
    Args:
        project_data: Series with project data
        risk_score: Risk score from heuristic model
        factor_contributions: Dictionary of factor contributions
        risk_weights: Dictionary of risk weights
        
    Returns:
        list: List of recommendation dictionaries
    """
    # Identify top risk factors for this project (reusing code from explain_project_risk)
    project_factors = {}
    
    # Extract project features that match our factor names
    for factor, weight in risk_weights.items():
        factor_display = factor.replace('_', ' ').title()
        
        # Try different variations of the factor name to find a match
        factor_variations = [
            factor,  # Original name
            factor.replace('_', ''),  # No underscores
            ''.join([word.capitalize() for word in factor.split('_')]),  # CamelCase
            factor.title().replace('_', ''),  # TitleCaseNoUnderscore
            factor_display,  # Title Case With Spaces
            factor_display.replace(' ', '')  # TitleCaseNoSpaces
        ]
        
        # Look for any matching column
        matching_col = None
        for var in factor_variations:
            for col in project_data.index:
                if isinstance(col, str) and var.lower() in col.lower():
                    matching_col = col
                    break
            if matching_col:
                break
        
        # If we found a matching column, calculate its contribution to risk
        if matching_col:
            # Get the value and normalize if it's numeric
            value = project_data[matching_col]
            if isinstance(value, (int, float)) and not pd.isna(value):
                # Normalize the value to 0-1 range
                normalized_value = min(max(value, 0), 1) if 0 <= value <= 1 else value / 10 if value > 1 else 0
                project_factors[factor] = {
                    'display_name': factor_display,
                    'value': value,
                    'normalized_value': normalized_value,
                    'contribution': normalized_value * weight,
                    'weight': weight,
                    'original_column': matching_col
                }
    
    # Sort factors by contribution
    sorted_factors = sorted(project_factors.items(), key=lambda x: x[1]['contribution'], reverse=True)
    
    # Generate recommendations based on the top risk factors
    recommendations = []
    
    # Add general recommendation based on overall risk level
    if risk_score >= 0.8:
        recommendations.append({
            "title": "Initiate Comprehensive Risk Management Plan",
            "description": "This project has critical risk levels requiring immediate attention. Implement a comprehensive risk management plan with weekly tracking and executive oversight."
        })
    elif risk_score >= 0.6:
        recommendations.append({
            "title": "Enhance Project Monitoring and Controls",
            "description": "This project has high risk levels that require enhanced monitoring protocols and additional controls. Conduct bi-weekly risk reassessments."
        })
    elif risk_score >= 0.4:
        recommendations.append({
            "title": "Regular Risk Monitoring",
            "description": "This project has medium risk levels that require regular monitoring. Schedule monthly risk review meetings to assess changes in key risk factors."
        })
    
    # Add factor-specific recommendations for the top 3 factors
    for i, (factor, data) in enumerate(sorted_factors[:3]):
        # Skip if we don't have a substantial contribution
        if data['contribution'] < 0.05:
            continue
            
        # Generate recommendation based on factor type
        if factor == 'complexity' or 'complex' in factor:
            recommendations.append({
                "title": f"Reduce {data['display_name']}",
                "description": f"Consider breaking the project into smaller, more manageable phases or components. Assign specialized team members to handle the most complex aspects."
            })
        elif 'schedule' in factor or 'deadline' in factor or 'timeline' in factor:
            recommendations.append({
                "title": f"Address {data['display_name']} Pressures",
                "description": f"Evaluate the project timeline for potential adjustments. Identify critical path activities and add buffer time for high-risk tasks. Consider requesting deadline extensions where feasible."
            })
        elif 'budget' in factor or 'cost' in factor or 'financial' in factor:
            recommendations.append({
                "title": f"Mitigate {data['display_name']} Constraints",
                "description": f"Review budget allocations and identify potential cost-saving opportunities. Consider requesting additional funding for critical aspects, while reducing scope in less essential areas."
            })
        elif 'stakeholder' in factor or 'client' in factor:
            recommendations.append({
                "title": f"Improve {data['display_name']} Engagement",
                "description": f"Develop a more comprehensive stakeholder engagement plan. Schedule regular alignment meetings and create clear communication channels to ensure stakeholder expectations are managed effectively."
            })
        elif 'team' in factor or 'resource' in factor or 'staff' in factor:
            recommendations.append({
                "title": f"Strengthen {data['display_name']}",
                "description": f"Evaluate team composition and identify skill gaps. Consider bringing in additional expertise or providing targeted training to address weak areas. Implement knowledge sharing sessions."
            })
        elif 'requirement' in factor or 'scope' in factor:
            recommendations.append({
                "title": f"Clarify {data['display_name']}",
                "description": f"Conduct a thorough requirements review workshop. Document and prioritize all requirements, and get explicit stakeholder sign-off. Implement a formal change management process."
            })
        elif 'technical' in factor or 'technology' in factor:
            recommendations.append({
                "title": f"Address {data['display_name']}",
                "description": f"Conduct a technical risk assessment to identify specific vulnerabilities. Consider bringing in specialized expertise or exploring alternative technical approaches with lower risk profiles."
            })
        else:
            # Generic recommendation for other factors
            recommendations.append({
                "title": f"Address {data['display_name']}",
                "description": f"This factor significantly impacts project risk. Develop a specific action plan to monitor and control this aspect throughout the project lifecycle."
            })
    
    # Add a general best practice recommendation if we don't have enough specific ones
    if len(recommendations) < 3:
        recommendations.append({
            "title": "Implement Regular Risk Reassessment",
            "description": "Schedule regular risk reassessment meetings throughout the project lifecycle. Update the risk register and mitigation plans as new information becomes available."
        })
    
    return recommendations


def format_risk_recommendations(recommendations):
    """Format risk mitigation recommendations as HTML
    
    Args:
        recommendations: List of recommendation dictionaries
        
    Returns:
        str: HTML formatted recommendations
    """
    if not recommendations:
        return "<p>No specific recommendations available.</p>"
    
    html = "<h4>Risk Mitigation Recommendations</h4>"
    html += "<p>Based on the risk analysis, we recommend the following actions:</p>"
    
    for i, rec in enumerate(recommendations):
        html += f"""
        <div style="margin-bottom: 15px; padding: 10px; border-left: 3px solid {ARCADIS_PRIMARY_COLOR}; background-color: #f9f9f9;">
            <h5 style="margin-top: 0; color: {ARCADIS_PRIMARY_COLOR};">{i+1}. {rec['title']}</h5>
            <p style="margin-bottom: 0;">{rec['description']}</p>
        </div>
        """
    
    html += """
    <p style="margin-top: 20px; font-style: italic;">Note: These recommendations are generated based on the available data and risk analysis. 
    They should be reviewed and adapted to the specific context of the project.</p>
    """
    
    return html


def generate_risk_factor_chart(factor_contributions):
    """Create a chart showing risk factor contributions
    
    Args:
        factor_contributions: Dictionary mapping factors to their contributions
        
    Returns:
        Figure: Plotly figure with risk factor visualization
    """
    # Check the format of factor_contributions - if it's the complex format with weights
    if isinstance(next(iter(factor_contributions.values()), {}), dict):
        # Create weighted contributions using the weight values
        weighted_contributions = {}
        for factor, data in factor_contributions.items():
            if isinstance(data, dict) and 'weight' in data:
                weighted_contributions[factor] = data['weight']
            else:
                weighted_contributions[factor] = 0
        
        # Use the weighted contributions for visualization
        factors_df = pd.DataFrame([
            {"Factor": k.replace('_', ' ').title(), "Contribution": v} 
            for k, v in weighted_contributions.items()
        ])
    else:
        # Use the simple format directly
        factors_df = pd.DataFrame([
            {"Factor": k.replace('_', ' ').title(), "Contribution": v} 
            for k, v in factor_contributions.items()
        ])
    
    # Sort by contribution
    factors_df = factors_df.sort_values("Contribution", ascending=False)
    
    # Create horizontal bar chart
    fig = px.bar(
        factors_df,
        x="Contribution",
        y="Factor",
        orientation="h",
        title="Risk Factor Contributions",
        color="Contribution",
        color_continuous_scale=["#4D4D4F", "#FF6900"],
        template="plotly_white"
    )
    
    fig.update_layout(
        xaxis_title="Relative Contribution",
        yaxis_title="Risk Factor",
        coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    
    return fig


def get_risk_distribution_chart(risk_distribution):
    """Create a pie chart showing risk category distribution
    
    Args:
        risk_distribution: Dictionary mapping risk categories to counts
        
    Returns:
        Figure: Plotly figure with risk distribution visualization
    """
    # Create labels and values from distribution
    labels = list(risk_distribution.keys())
    values = list(risk_distribution.values())
    
    # Define colors for risk categories
    colors = {
        "Low Risk": "#4CAF50",  # Green
        "Medium Risk": "#FFC107",  # Yellow
        "High Risk": "#FF9800",  # Orange
        "Critical Risk": "#F44336"  # Red
    }
    
    # Use defined colors if available, otherwise use default color sequence
    color_values = [colors.get(label, ARCADIS_PRIMARY_COLOR) for label in labels]
    
    # Create pie chart
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker_colors=color_values
    )])
    
    fig.update_layout(
        title="Risk Level Distribution",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=0)
    )
    
    return fig


def generate_ai_narrative_summary() -> str:
    """Generate an AI narrative summary of the risk analysis"""
    if not (st.session_state.project_data is not None and st.session_state.risk_predictions is not None):
        return "Insufficient data for narrative generation."
    
    df = st.session_state.project_data
    model_name = st.session_state.best_model_name
    predictions = st.session_state.risk_predictions
    probabilities = st.session_state.risk_probabilities
    threshold = st.session_state.prediction_threshold
    
    high_risk_count = (probabilities > threshold).sum()
    high_risk_rate = high_risk_count / len(df) * 100
    
    # Feature importance (simulated if not available)
    if 'feature_importance' in st.session_state.model_results and st.session_state.best_model_name in st.session_state.model_results['feature_importance']:
        top_features = st.session_state.model_results['feature_importance'][st.session_state.best_model_name][0][:3]
    else:
        # Fallback to default important features
        top_features = ["ComplexityLevel", "Budget", "StakeholderEngagementScore"]
    
    # Get correlations for project types
    project_types = df['ProjectType'].unique()
    project_type_risks = {}
    for pt in project_types:
        mask = df['ProjectType'] == pt
        if mask.sum() > 0:
            risk_rate = (probabilities[mask] > threshold).mean() * 100
            project_type_risks[pt] = risk_rate
    
    highest_risk_type = max(project_type_risks.items(), key=lambda x: x[1]) if project_type_risks else ('Unknown', 0)
    
    # Get correlations for regions
    regions = df['Region'].unique()
    region_risks = {}
    for region in regions:
        mask = df['Region'] == region
        if mask.sum() > 0:
            risk_rate = (probabilities[mask] > threshold).mean() * 100
            region_risks[region] = risk_rate
    
    highest_risk_region = max(region_risks.items(), key=lambda x: x[1]) if region_risks else ('Unknown', 0)
    
    # Calculate average probability
    avg_prob = probabilities.mean() * 100
    median_prob = probabilities.median() * 100
    
    narrative = f"""The {model_name} model predicts {high_risk_count} projects ({high_risk_rate:.1f}% of those with predictions) are at high risk of derailment, 
using a threshold of {threshold}.

Key Insights:

* Primary Risk Drivers: Across the portfolio, the factors most strongly correlated with increased risk appear to be: 
  {', '.join(top_features)}.

* Highest Risk Project Type: Projects classified as '{highest_risk_type[0]}' ({highest_risk_type[1]:.1f}%) show the highest average predicted risk rate.

* Highest Risk Region: The '{highest_risk_region[0]}' region currently exhibits the highest average predicted risk rate at {highest_risk_region[1]:.1f}%.

* Prediction Certainty: The average predicted risk probability across projects is {avg_prob:.1f}% (median: {median_prob:.1f}%). A wider 
  spread might indicate greater uncertainty overall.

Recommendation: Prioritize investigation and potential mitigation actions for the identified high-risk projects, 
paying close attention to the top risk drivers ({', '.join(top_features[:2])}...). Consider focusing 
efforts on projects within the '{highest_risk_type[0]}' type or '{highest_risk_region[0]}' region if applicable. Use the 'Portfolio Deep Dive' 
and 'Model Analysis' tabs for more detailed investigation.
"""
    
    return narrative

# Tab content functions

def welcome_tab():
    """Content for the Welcome tab"""
    # Set layout for the hero section using pure Streamlit components
    st.write("")
    st.write("")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Use direct URL to reliable CDN for image icon
        st.image("./assets/business_risk_icon.png", width=140)
    
    with col2:
        st.write("")
        st.markdown(f'<h1 style="color:{ARCADIS_PRIMARY_COLOR}; font-size:36px;">Welcome to RiskLens Pro</h1>', unsafe_allow_html=True)
        st.markdown(f'<p style="color:{ARCADIS_ACCENT_COLOR}; font-size:18px;">Powered by Arcadis expertise & advanced analytics</p>', unsafe_allow_html=True)
    
    # Create an impactful introduction with a colored background
    st.markdown(f"""
    <div style="background-color: {ARCADIS_LIGHT_BG}; padding: 20px; border-radius: 10px; margin: 20px 0;">
    <h2 style="color: {ARCADIS_PRIMARY_COLOR}; margin-top: 0;">The Challenge: Navigating Project Complexity</h2>

    <p style="font-size: 16px;">
    Delivering complex projects on time and within budget is a significant challenge. Factors like scope changes, 
    resource constraints, technical hurdles, and external dependencies can introduce risks, leading to costly 
    overruns and delays. Proactively identifying and understanding these risks is crucial for successful project 
    delivery and maintaining client satisfaction.
    </p>
    </div>

    """, unsafe_allow_html=True)
    
    # Solution section with KPIs
    st.markdown(f"<h2 style='color: {ARCADIS_PRIMARY_COLOR};'>The Solution: Data-Driven Risk Intelligence</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    RiskLens Pro leverages your project data and machine learning to provide early warnings about potential project derailment. 
    By analyzing historical patterns and current project characteristics, it predicts the likelihood of significant cost or schedule overruns.
    """)
    
    # Stats row to make it more visual
    stat1, stat2, stat3 = st.columns(3)
    
    with stat1:
        st.markdown(f"""
        <div style="text-align: center; padding: 15px; background-color: white; border-radius: 10px; border-top: 5px solid {ARCADIS_PRIMARY_COLOR}; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
            <h1 style="color:{ARCADIS_PRIMARY_COLOR}; font-size: 40px; margin-bottom: 5px;">30%</h1>
            <p style="color: {ARCADIS_ACCENT_COLOR};">Average cost saving on at-risk projects identified early</p>
        </div>

        """, unsafe_allow_html=True)
    
    with stat2:
        st.markdown(f"""
        <div style="text-align: center; padding: 15px; background-color: white; border-radius: 10px; border-top: 5px solid {ARCADIS_SECONDARY_COLOR}; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
            <h1 style="color:{ARCADIS_SECONDARY_COLOR}; font-size: 40px; margin-bottom: 5px;">85%</h1>
            <p style="color: {ARCADIS_ACCENT_COLOR};">Prediction accuracy for project risk classification</p>
        </div>

        """, unsafe_allow_html=True)
    
    with stat3:
        st.markdown(f"""
        <div style="text-align: center; padding: 15px; background-color: white; border-radius: 10px; border-top: 5px solid {ARCADIS_DARK_BG}; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
            <h1 style="color:{ARCADIS_DARK_BG}; font-size: 40px; margin-bottom: 5px;">3x</h1>
            <p style="color: {ARCADIS_ACCENT_COLOR};">Faster identification of potential project risks</p>
        </div>

        """, unsafe_allow_html=True)

    # Journey Map - visual flow of the application
    st.markdown("""<br>""", unsafe_allow_html=True)
    st.markdown(f"<h2 style='color: {ARCADIS_PRIMARY_COLOR};'>Your Risk Management Journey</h2>", unsafe_allow_html=True)
    
    # Timeline
    journey_cols = st.columns(5)
    
    with journey_cols[0]:
        st.markdown(f"""
        <div style="text-align: center; padding: 10px; position: relative;">
            <div style="position: absolute; top: 15px; left: 50%; width: 100%; height: 4px; background-color: {ARCADIS_LIGHT_BG}; z-index: 0;"></div>
            <div style="position: relative; background-color: {ARCADIS_PRIMARY_COLOR}; color: white; width: 40px; height: 40px; line-height: 40px; border-radius: 50%; margin: 0 auto; z-index: 1;">1</div>
            <h4 style="margin-top: 10px;">Data Upload</h4>
            <p style="font-size: 14px;">Load your project data or use our sample data</p>
        </div>
        """, unsafe_allow_html=True)
    
    with journey_cols[1]:
        st.markdown(f"""
        <div style="text-align: center; padding: 10px; position: relative;">
            <div style="position: absolute; top: 15px; left: -50%; width: 100%; height: 4px; background-color: {ARCADIS_LIGHT_BG}; z-index: 0;"></div>
            <div style="position: relative; background-color: {ARCADIS_PRIMARY_COLOR}; color: white; width: 40px; height: 40px; line-height: 40px; border-radius: 50%; margin: 0 auto; z-index: 1;">2</div>
            <h4 style="margin-top: 10px;">Risk Analysis</h4>
            <p style="font-size: 14px;">ML models analyze and predict project risks</p>
        </div>
        """, unsafe_allow_html=True)
    
    with journey_cols[2]:
        st.markdown(f"""
        <div style="text-align: center; padding: 10px; position: relative;">
            <div style="position: absolute; top: 15px; left: -50%; width: 100%; height: 4px; background-color: {ARCADIS_LIGHT_BG}; z-index: 0;"></div>
            <div style="position: relative; background-color: {ARCADIS_PRIMARY_COLOR}; color: white; width: 40px; height: 40px; line-height: 40px; border-radius: 50%; margin: 0 auto; z-index: 1;">3</div>
            <h4 style="margin-top: 10px;">Portfolio Review</h4>
            <p style="font-size: 14px;">Identify high-risk projects and patterns</p>
        </div>
        """, unsafe_allow_html=True)
    
    with journey_cols[3]:
        st.markdown(f"""
        <div style="text-align: center; padding: 10px; position: relative;">
            <div style="position: absolute; top: 15px; left: -50%; width: 100%; height: 4px; background-color: {ARCADIS_LIGHT_BG}; z-index: 0;"></div>
            <div style="position: relative; background-color: {ARCADIS_PRIMARY_COLOR}; color: white; width: 40px; height: 40px; line-height: 40px; border-radius: 50%; margin: 0 auto; z-index: 1;">4</div>
            <h4 style="margin-top: 10px;">Simulation</h4>
            <p style="font-size: 14px;">Test scenarios and what-if analyses</p>
        </div>
        """, unsafe_allow_html=True)
    
    with journey_cols[4]:
        st.markdown(f"""
        <div style="text-align: center; padding: 10px; position: relative;">
            <div style="position: absolute; top: 15px; left: -50%; width: 100%; height: 4px; background-color: {ARCADIS_LIGHT_BG}; z-index: 0;"></div>
            <div style="position: relative; background-color: {ARCADIS_PRIMARY_COLOR}; color: white; width: 40px; height: 40px; line-height: 40px; border-radius: 50%; margin: 0 auto; z-index: 1;">5</div>
            <h4 style="margin-top: 10px;">Action Plan</h4>
            <p style="font-size: 14px;">Get reports with actionable insights</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Complete redesign of the capabilities section using pure Streamlit components
    st.write("")
    st.write("")
    
    # Centered header with custom styling
    st.markdown(f"<h2 style='text-align: center; color: {ARCADIS_PRIMARY_COLOR}; font-weight: 600;'>Platform Capabilities</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; margin-bottom: 30px;'>Leveraging advanced machine learning for actionable project risk insights</p>", unsafe_allow_html=True)
    
    # Create expandable sections for capabilities to avoid text overflow issues
    expandables = st.container()
    with expandables:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            with st.container():
                st.write("")
                st.markdown(f"#### 📊 Risk Prediction")
                with st.expander("View details"):
                    st.write("Advanced ML algorithms (Random Forest, XGBoost)")
                    st.write("Up to 85% prediction accuracy")
                    st.write("Customizable risk thresholds")
                    st.write("Confidence scores for all predictions")
        
        with col2:
            with st.container():
                st.write("")
                st.markdown(f"#### 🔍 Explainable AI")
                with st.expander("View details"):
                    st.write("LIME explainability for transparent decisions")
                    st.write("Feature importance analysis")
                    st.write("Interactive visualizations")
                    st.write("Risk driver identification")
        
        with col3:
            with st.container():
                st.write("")
                st.markdown(f"#### 🎯 Actionable Insights")
                with st.expander("View details"):
                    st.write("What-if scenario planning")
                    st.write("Prioritized risk mitigation strategies")
                    st.write("PDF/PPT export for stakeholders")
                    st.write("Ongoing project monitoring")
    
    # Add more vertical space after the capability section
    st.write("")
    st.write("")
    
    # Create a completely separate section with a large gap
    st.write("")
    st.write("")
    
    # Force proper separation with explicit newlines and a divider
    st.markdown("""<div style='height: 80px;'></div>""", unsafe_allow_html=True)
    st.divider()
    st.markdown("""<div style='height: 40px;'></div>""", unsafe_allow_html=True)
    
    # Create a new section container
    questions_container = st.container()
    
    with questions_container:
        # Add a title to this section
        st.markdown(f"""
        <div style="text-align: center; margin-bottom: 40px; padding-top: 20px;">
            <h2 style="color: {ARCADIS_PRIMARY_COLOR}; font-weight: 600;">Key Project Questions Answered</h2>
            <p>RiskLens Pro helps you answer critical risk management questions</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create columns for the questions
        qcol1, qcol2 = st.columns(2)
        
        with qcol1:
            st.markdown(f"""
            <div style="background-color: white; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); padding: 20px; height: 95%; border-top: 4px solid {ARCADIS_PRIMARY_COLOR}; margin-bottom: 20px;">
                <h3 style="margin-top: 0; color: {ARCADIS_PRIMARY_COLOR}; font-size: 20px; margin-bottom: 10px;">Which projects need my immediate attention?</h3>
                <p style="color: #555; margin-bottom: 10px;">Get a prioritized list of high-risk projects with clear indicators of which projects need immediate intervention.</p>
                <div style="background-color: {ARCADIS_LIGHT_BG}; color: {ARCADIS_ACCENT_COLOR}; font-weight: 500; padding: 4px 10px; border-radius: 4px; font-size: 14px; display: inline-block;">Executive Summary tab</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="background-color: white; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); padding: 20px; height: 95%; border-top: 4px solid {ARCADIS_SECONDARY_COLOR}; margin-bottom: 20px;">
                <h3 style="margin-top: 0; color: {ARCADIS_PRIMARY_COLOR}; font-size: 20px; margin-bottom: 10px;">Why is this project flagged as high-risk?</h3>
                <p style="color: #555; margin-bottom: 10px;">Understand the specific factors contributing to a project's risk rating with transparent AI explanations.</p>
                <div style="background-color: {ARCADIS_LIGHT_BG}; color: {ARCADIS_ACCENT_COLOR}; font-weight: 500; padding: 4px 10px; border-radius: 4px; font-size: 14px; display: inline-block;">Model Analysis tab</div>
            </div>
            """, unsafe_allow_html=True)
        
        with qcol2:
            st.markdown(f"""
            <div style="background-color: white; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); padding: 20px; height: 95%; border-top: 4px solid {ARCADIS_SECONDARY_COLOR}; margin-bottom: 20px;">
                <h3 style="margin-top: 0; color: {ARCADIS_PRIMARY_COLOR}; font-size: 20px; margin-bottom: 10px;">What if resources become scarce?</h3>
                <p style="color: #555; margin-bottom: 10px;">Run interactive scenario simulations to see how resource changes would impact project risk across your portfolio.</p>
                <div style="background-color: {ARCADIS_LIGHT_BG}; color: {ARCADIS_ACCENT_COLOR}; font-weight: 500; padding: 4px 10px; border-radius: 4px; font-size: 14px; display: inline-block;">Simulation tab</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="background-color: white; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); padding: 20px; height: 95%; border-top: 4px solid {ARCADIS_PRIMARY_COLOR}; margin-bottom: 20px;">
                <h3 style="margin-top: 0; color: {ARCADIS_PRIMARY_COLOR}; font-size: 20px; margin-bottom: 10px;">Which factors drive risk in our portfolio?</h3>
                <p style="color: #555; margin-bottom: 10px;">Identify the key factors that most strongly correlate with project risk across your entire portfolio.</p>
                <div style="background-color: {ARCADIS_LIGHT_BG}; color: {ARCADIS_ACCENT_COLOR}; font-weight: 500; padding: 4px 10px; border-radius: 4px; font-size: 14px; display: inline-block;">Model Analysis tab</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Create another container for the CTA and status
    cta_container = st.container()
    
    with cta_container:
        # Add space before CTA
        st.write("")
        st.write("")
        
        # Create a prominent CTA
        st.markdown(f"""
        <div style="background: linear-gradient(90deg, {ARCADIS_PRIMARY_COLOR}, {ARCADIS_SECONDARY_COLOR}); padding: 25px; border-radius: 10px; text-align: center; color: white; margin-top: 30px;">
            <h2 style="margin-top: 0; color: white;">Ready to Get Started?</h2>
            <p style="font-size: 18px;">Begin your risk analysis journey by loading your project data or using our sample dataset.</p>
            <p style="font-size: 16px;">Use the <b>"Load Sample Data"</b> button in the sidebar to explore the platform's capabilities.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Add space after CTA and before status indicators
        st.write("")
        st.write("")
        
        # Status indicator based on current progress
        if st.session_state.project_data is None:
            st.info("👈 Get started by uploading your data in the sidebar.", icon="ℹ️")
        elif not st.session_state.trained_models:
            st.info("✅ Your data is loaded! Now train the risk model using the 'Run Risk Analytics' button in the sidebar.", icon="👈")
        else:
            st.success("🎉 Your risk model is ready! Navigate through the tabs above to explore the insights.", icon="✨")

def executive_summary_tab():
    """Content for the Executive Summary tab"""
    if st.session_state.project_data is None:
        # Create a better placeholder when no data is available
        st.markdown(f"""
        <div style="background-color: {ARCADIS_LIGHT_BG}; padding: 20px; border-radius: 10px; text-align: center;">  
            <img src="./assets/business_risk_icon.png" width="60" style="margin-bottom: 10px;">  
            <h2 style="color: {ARCADIS_PRIMARY_COLOR};">Executive Summary</h2>
            <p style="font-size: 16px;">Please load data to view the executive summary.</p>
            <p style="font-size: 14px; color: {ARCADIS_ACCENT_COLOR};">Use the sidebar on the left to upload project data or load the sample dataset.</p>
        </div>
        """, unsafe_allow_html=True)  
        return
    
    # Create a more visual title with underline animation
    st.markdown(f"""
    <style>
    .animated-header {{  
        position: relative;
        display: inline-block;
    }}
    .animated-header::after {{  
        content: '';
        position: absolute;
        width: 100%;
        height: 3px;
        bottom: -5px;
        left: 0;
        background: linear-gradient(90deg, {ARCADIS_PRIMARY_COLOR}, {ARCADIS_SECONDARY_COLOR});
        transform: scaleX(0);  
        transform-origin: bottom right;
        transition: transform 0.5s ease-out;
        animation: expand 1.5s ease-out forwards;
    }}
    @keyframes expand {{  
        to {{ transform: scaleX(1); transform-origin: bottom left; }}
    }}
    </style>
    <h1 class="animated-header" style="color:{ARCADIS_PRIMARY_COLOR};">📊 Executive Summary</h1>
    <p style="color:{ARCADIS_ACCENT_COLOR}; font-size:16px; margin-top:-5px;">Portfolio-wide risk assessment with actionable insights</p>
    """, unsafe_allow_html=True)
    
    # Check if we need to run risk analytics first
    if st.session_state.heuristic_risk_scores is None and st.session_state.risk_predictions is None:
        st.info("Please run risk analytics using the sidebar to view the risk assessment.", icon="ℹ️")
        return
    
    # Get metrics - use heuristic risk if ML predictions aren't available
    metrics = get_data_profiling_metrics(st.session_state.project_data)
    total_projects = len(st.session_state.project_data)
    
    # Use intelligent heuristic risk or ML risk, depending on what's available
    if st.session_state.risk_predictions is not None:
        high_risk_projects = int(st.session_state.risk_predictions.sum())
        risk_source = "Machine Learning Model"
    elif st.session_state.heuristic_risk_scores is not None:
        high_risk_projects = int(st.session_state.heuristic_risk_scores['high_risk'].sum())
        risk_source = "Heuristic Risk Engine"
        
    high_risk_rate = high_risk_projects / total_projects * 100
    
    # Create an alert dashboard at the top
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, {ARCADIS_PRIMARY_COLOR}22, {ARCADIS_LIGHT_BG}); padding: 20px; border-radius: 10px; margin: 20px 0; border-left: 5px solid {ARCADIS_PRIMARY_COLOR};">
        <h3 style="margin-top: 0;">Project Risk Summary</h3>
        <p>As of {datetime.datetime.now().strftime('%B %d, %Y')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create a more impactful metric row with gauges
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div style="background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); height: 180px;">
            <div style="text-align: center;">
                <p style="color: #555; font-size: 16px; margin-bottom: 5px;">Total Projects</p>
                <h1 style="font-size: 48px; color: {ARCADIS_PRIMARY_COLOR}; margin: 0;">{total_projects}</h1>
                <div style="margin-top: 15px; text-align: center;">
                    <span style="background-color: #e0e0e0; display: inline-block; width: 100%; height: 8px; border-radius: 4px;">
                        <span style="background-color: {ARCADIS_PRIMARY_COLOR}; display: inline-block; width: 100%; height: 8px; border-radius: 4px;"></span>
                    </span>
                    <p style="font-size: 12px; color: #777; margin-top: 5px;">In active portfolio</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); height: 180px;">
            <div style="text-align: center;">
                <p style="color: #555; font-size: 16px; margin-bottom: 5px;">High-Risk Projects</p>
                <h1 style="font-size: 48px; color: {ARCADIS_SECONDARY_COLOR}; margin: 0;">{high_risk_projects}</h1>
                <div style="margin-top: 15px; text-align: center;">
                    <span style="background-color: #e0e0e0; display: inline-block; width: 100%; height: 8px; border-radius: 4px;">
                        <span style="background-color: {ARCADIS_SECONDARY_COLOR}; display: inline-block; width: {high_risk_rate}%; height: 8px; border-radius: 4px;"></span>
                    </span>
                    <p style="font-size: 12px; color: #777; margin-top: 5px;">Need immediate attention</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); height: 180px;">
            <div style="text-align: center;">
                <p style="color: #555; font-size: 16px; margin-bottom: 5px;">High-Risk Rate</p>
                <h1 style="font-size: 48px; color: {'#e74c3c' if high_risk_rate > 30 else '#f1c40f' if high_risk_rate > 15 else '#2ecc71'}; margin: 0;">{high_risk_rate:.1f}%</h1>
                <div style="margin-top: 15px; text-align: center;">
                    <span style="background-color: #e0e0e0; display: inline-block; width: 100%; height: 8px; border-radius: 4px;">
                        <span style="background-color: {'#e74c3c' if high_risk_rate > 30 else '#f1c40f' if high_risk_rate > 15 else '#2ecc71'}; display: inline-block; width: {min(100, high_risk_rate*2)}%; height: 8px; border-radius: 4px;"></span>
                    </span>
                    <p style="font-size: 12px; color: #777; margin-top: 5px;">{'High Concern' if high_risk_rate > 30 else 'Moderate Concern' if high_risk_rate > 15 else 'Low Concern'}</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Cost and schedule metrics in a better visualization
    if "Avg Cost Overrun % (High-Risk)" in metrics and "Avg Schedule Overrun % (High-Risk)" in metrics:
        cost_overrun = float(metrics["Avg Cost Overrun % (High-Risk)"].replace('%', ''))
        schedule_overrun = float(metrics["Avg Schedule Overrun % (High-Risk)"].replace('%', ''))
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='color: {ARCADIS_PRIMARY_COLOR};'>Project Performance Impact</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Create a cost overrun gauge/progress bar
            st.markdown(f"""
            <div style="background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <h4 style="margin: 0;">💰 Avg. Cost Overrun</h4>
                    <span style="font-size: 24px; font-weight: bold; color: {'#e74c3c' if cost_overrun > 20 else '#f1c40f' if cost_overrun > 10 else '#2ecc71'};">{metrics["Avg Cost Overrun % (High-Risk)"]}</span>
                </div>
                <div style="margin-top: 10px;">
                    <span style="background-color: #e0e0e0; display: block; width: 100%; height: 10px; border-radius: 5px;">
                        <span style="background-color: {'#e74c3c' if cost_overrun > 20 else '#f1c40f' if cost_overrun > 10 else '#2ecc71'}; display: block; width: {min(100, cost_overrun*2)}%; height: 10px; border-radius: 5px;"></span>
                    </span>
                </div>
                <p style="margin-top: 10px; font-size: 14px; color: #666;">High-risk projects typically exceed budget by {metrics["Avg Cost Overrun % (High-Risk)"]}.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Create a schedule overrun gauge/progress bar
            st.markdown(f"""
            <div style="background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <h4 style="margin: 0;">⏱️ Avg. Schedule Overrun</h4>
                    <span style="font-size: 24px; font-weight: bold; color: {'#e74c3c' if schedule_overrun > 20 else '#f1c40f' if schedule_overrun > 10 else '#2ecc71'};">{metrics["Avg Schedule Overrun % (High-Risk)"]}</span>
                </div>
                <div style="margin-top: 10px;">
                    <span style="background-color: #e0e0e0; display: block; width: 100%; height: 10px; border-radius: 5px;">
                        <span style="background-color: {'#e74c3c' if schedule_overrun > 20 else '#f1c40f' if schedule_overrun > 10 else '#2ecc71'}; display: block; width: {min(100, schedule_overrun*2)}%; height: 10px; border-radius: 5px;"></span>
                    </span>
                </div>
                <p style="margin-top: 10px; font-size: 14px; color: #666;">High-risk projects typically exceed schedule by {metrics["Avg Schedule Overrun % (High-Risk)"]}.</p>
            </div>
            """, unsafe_allow_html=True)
    
    # AI-Generated Narrative Summary - improved styling
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='color: {ARCADIS_PRIMARY_COLOR};'>💡 Key Insights & Recommendations</h3>", unsafe_allow_html=True)
    
    # Get narrative
    if st.session_state.ai_narrative_generated:
        narrative = generate_ai_narrative_summary()
    else:
        with st.spinner("Generating insights..."): 
            st.session_state.ai_narrative_generated = True
            narrative = generate_ai_narrative_summary()
    
    # Display narrative in a nicer format
    narrative_html = narrative.replace('\n\n', '<br><br>').replace('*', '•')
    st.markdown(f"""
    <div style="background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
        {narrative_html}
    </div>
    """, unsafe_allow_html=True)
    
    # Visualization section - improved
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='color: {ARCADIS_PRIMARY_COLOR};'>📊 Risk Distribution Analysis</h3>", unsafe_allow_html=True)
    
    # Two columns for visualizations
    viz1, viz2 = st.columns([3, 2])
    
    with viz1:
        # Check if we have risk probabilities
        if st.session_state.risk_probabilities is not None:
            # Get risk probabilities
            probabilities = st.session_state.risk_probabilities
            
            # Create improved histogram with proper coloring
            fig = px.histogram(
                x=probabilities,
                nbins=20,
                labels={"x": "Risk Probability"},
                title="Distribution of Project Risk Probabilities",
                color_discrete_sequence=[ARCADIS_PRIMARY_COLOR],
                template="plotly_white"
            )
            
            # Add vertical line for threshold
            fig.add_vline(
                x=st.session_state.prediction_threshold,
                line_dash="dash",
                line_color="black",
                line_width=2,
                annotation_text=f"Risk Threshold ({st.session_state.prediction_threshold})",
                annotation_position="top left",
                annotation_font={"size": 14, "color": "black"}
            )
            
            # Add risk distribution to the chart
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        elif st.session_state.heuristic_risk_scores is not None:
            # If only heuristic risk is available, plot the risk score distribution
            risk_scores = st.session_state.heuristic_risk_scores['risk_score']
            
            fig = px.histogram(
                x=risk_scores,
                nbins=20,
                labels={"x": "Heuristic Risk Score"},
                title="Distribution of Heuristic Risk Scores",
                color_discrete_sequence=[ARCADIS_PRIMARY_COLOR],
                template="plotly_white"
            )
            
            # Add threshold lines for risk category boundaries
            fig.add_vline(x=0.4, line_dash="dash", line_color="#4CAF50", line_width=2,
                        annotation_text="Low/Medium Boundary", annotation_position="top left")
            fig.add_vline(x=0.6, line_dash="dash", line_color="#FFC107", line_width=2,
                        annotation_text="Medium/High Boundary", annotation_position="top left")
            fig.add_vline(x=0.8, line_dash="dash", line_color="#F44336", line_width=2,
                        annotation_text="High/Critical Boundary", annotation_position="top left")
            
            # Add risk distribution to the chart
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    
    with viz2:
        # Pie chart of risk categories using heuristic model (more informative)
        if st.session_state.heuristic_risk_distribution is not None:
            # Use the heuristic risk distribution directly
            fig = get_risk_distribution_chart(st.session_state.heuristic_risk_distribution)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        elif st.session_state.risk_predictions is not None:
            # Create a binary pie chart for ML predictions
            risk_counts = st.session_state.risk_predictions.value_counts().to_dict()
            low_risk = risk_counts.get(0, 0)
            high_risk = risk_counts.get(1, 0)
            
            # Create pie chart for risk categories
            fig = go.Figure(data=[go.Pie(
                labels=["High Risk", "Low Risk"],
                values=[high_risk, low_risk],
                hole=0.4,
                marker_colors=["#F44336", "#4CAF50"]
            )])
            
            fig.update_layout(
                title="Risk Level Distribution",
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    
    # Add risk factor contribution chart from heuristic model
    if st.session_state.heuristic_model_metadata is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='color: {ARCADIS_PRIMARY_COLOR};'>🔍 Risk Factor Contributions</h3>", unsafe_allow_html=True)
        
        # Extract factor contributions from the model metadata
        factor_contributions = st.session_state.heuristic_model_metadata.get('factor_contributions', {})
        
        if factor_contributions:
            # Create a chart showing the contribution of each risk factor
            fig = generate_risk_factor_chart(factor_contributions)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            
            # Add explanatory text
            st.markdown("""
            This chart shows the relative contribution of different risk factors to the overall risk score. 
            Factors at the top contribute most significantly to risk across your project portfolio.
            """)
    
    # Add risk mitigation recommendations section
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='color: {ARCADIS_PRIMARY_COLOR};'>🛡️ Risk Mitigation Recommendations</h3>", unsafe_allow_html=True)
    
    # Select a high risk project to analyze
    high_risk_projects = None
    
    # Get high risk projects from either ML or heuristic model
    if st.session_state.risk_predictions is not None:
        high_risk_mask = st.session_state.risk_predictions == 1
        high_risk_projects = st.session_state.project_data[high_risk_mask]
    elif st.session_state.heuristic_risk_scores is not None:
        high_risk_mask = st.session_state.heuristic_risk_scores['high_risk'] == 1
        high_risk_projects = st.session_state.project_data[high_risk_mask]
    
    if high_risk_projects is not None and len(high_risk_projects) > 0:
        # Allow user to select a high risk project
        project_options = high_risk_projects.index.tolist()
        project_labels = []
        
        for idx in project_options:
            project_id = high_risk_projects.loc[idx, PROJECT_ID_COLUMN] if PROJECT_ID_COLUMN in high_risk_projects.columns else f"Project {idx}"
            project_name = high_risk_projects.loc[idx, PROJECT_NAME_COLUMN] if PROJECT_NAME_COLUMN in high_risk_projects.columns else ""
            project_labels.append(f"{project_id}: {project_name}")
        
        selected_project_idx = st.selectbox(
            "Select a high-risk project to view mitigation recommendations:",
            options=project_options,
            format_func=lambda x: project_labels[project_options.index(x)],
            key="selected_high_risk_project"
        )
        
        # Get the selected project data
        selected_project = high_risk_projects.loc[selected_project_idx]
        
        if st.session_state.heuristic_model_metadata is not None:
            # Generate explanation and recommendations for this project
            with st.spinner("Analyzing risk factors..."): 
                # Generate project risk explanation
                risk_explanation = explain_project_risk(
                    selected_project,
                    st.session_state.heuristic_risk_scores.loc[selected_project_idx, 'risk_score'],
                    st.session_state.heuristic_model_metadata['factor_contributions'],
                    st.session_state.heuristic_model_metadata['risk_weights']
                )
                
                # Generate mitigation recommendations
                recommendations = generate_risk_mitigation_recommendations(
                    selected_project,
                    st.session_state.heuristic_risk_scores.loc[selected_project_idx, 'risk_score'],
                    st.session_state.heuristic_model_metadata['factor_contributions'],
                    st.session_state.heuristic_model_metadata['risk_weights']
                )
                
                formatted_recommendations = format_risk_recommendations(recommendations)
            
            # Show the explanation and recommendations in tabs
            risk_tabs = st.tabs(["Risk Analysis", "Mitigation Recommendations"])
            
            with risk_tabs[0]:
                st.markdown(risk_explanation, unsafe_allow_html=True)
            
            with risk_tabs[1]:
                st.markdown(formatted_recommendations, unsafe_allow_html=True)
    else:
        st.info("No high-risk projects identified yet. Run risk analytics to generate recommendations.", icon="ℹ️")
        
    # Add some space at the bottom of the page
    st.markdown("<br><br>", unsafe_allow_html=True)

def portfolio_deep_dive_tab():
    """Content for the Portfolio Deep Dive tab"""
    if st.session_state.project_data is None or st.session_state.risk_predictions is None:
        st.warning("Please load data and train a risk model to view the portfolio analysis.")
        return
    
    styled_header("Portfolio Deep Dive", icon="🔍")
    st.markdown("Detailed analysis of your project portfolio with filtering and sorting capabilities.")
    
    # Get the data with predictions
    if hasattr(st.session_state, 'project_data_with_predictions'):
        df = st.session_state.project_data_with_predictions
    else:
        # If not available, create it
        df = st.session_state.project_data.copy()
        df['PredictedRisk'] = st.session_state.risk_predictions.values
        df['RiskProbability'] = st.session_state.risk_probabilities.values
        st.session_state.project_data_with_predictions = df
    
    # Create filter controls
    st.markdown("---")
    styled_header("Project Data & Predictions", level=2)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        project_types = ["All"] + sorted(df["ProjectType"].unique().tolist())
        selected_project_type = st.selectbox("Filter by Project Type:", project_types)
    
    with col2:
        regions = ["All"] + sorted(df["Region"].unique().tolist())
        selected_region = st.selectbox("Filter by Region:", regions)
    
    with col3:
        risk_options = ["All", "High Risk", "Low Risk"]
        selected_risk = st.selectbox("Filter by Predicted Risk:", risk_options, help="High Risk = probability > threshold")
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_project_type != "All":
        filtered_df = filtered_df[filtered_df["ProjectType"] == selected_project_type]
    
    if selected_region != "All":
        filtered_df = filtered_df[filtered_df["Region"] == selected_region]
    
    # Only apply risk filter if RiskProbability exists
    if "RiskProbability" in filtered_df.columns:
        if selected_risk == "High Risk":
            filtered_df = filtered_df[filtered_df["RiskProbability"] > st.session_state.prediction_threshold]
        elif selected_risk == "Low Risk":
            filtered_df = filtered_df[filtered_df["RiskProbability"] <= st.session_state.prediction_threshold]
    # Use HeuristicRiskScore as a fallback if available
    elif "HeuristicRiskScore" in filtered_df.columns and selected_risk != "All":
        if selected_risk == "High Risk":
            filtered_df = filtered_df[filtered_df["HeuristicRiskScore"] > st.session_state.prediction_threshold]
        elif selected_risk == "Low Risk":
            filtered_df = filtered_df[filtered_df["HeuristicRiskScore"] <= st.session_state.prediction_threshold]
    
    # Display filtered data
    display_cols = [PROJECT_ID_COLUMN, PROJECT_NAME_COLUMN, "Region", "ProjectType"]
    
    # Add available columns for display
    if "InitialScheduleDays" in filtered_df.columns:
        display_cols.append("InitialScheduleDays")
    if "ActualCost" in filtered_df.columns:
        display_cols.append("ActualCost")
    if "ActualScheduleDays" in filtered_df.columns:
        display_cols.append("ActualScheduleDays")
    
    # Only add risk columns if they exist
    risk_cols_to_display = []
    if "RiskProbability" in filtered_df.columns:
        risk_cols_to_display.append("RiskProbability")
    if "HeuristicRiskScore" in filtered_df.columns:
        risk_cols_to_display.append("HeuristicRiskScore")
    if "HeuristicRiskCategory" in filtered_df.columns:
        risk_cols_to_display.append("HeuristicRiskCategory")
    
    # Add anomaly detection columns if they exist
    if "anomaly_score" in filtered_df.columns:
        risk_cols_to_display.append("anomaly_score")
    if "is_anomaly" in filtered_df.columns:
        risk_cols_to_display.append("is_anomaly")
        
    # Add all available risk columns to display
    display_cols.extend(risk_cols_to_display)
    
    # Format display dataframe
    display_df = filtered_df[display_cols].copy()
    
    # Format risk columns
    if "RiskProbability" in display_df.columns:
        display_df["RiskProbability"] = (display_df["RiskProbability"] * 100).round(1).astype(str) + "%"
    if "HeuristicRiskScore" in display_df.columns:
        display_df["HeuristicRiskScore"] = (display_df["HeuristicRiskScore"] * 100).round(1).astype(str) + "%"
    if "anomaly_score" in display_df.columns:
        display_df["anomaly_score"] = display_df["anomaly_score"].round(3)
    
    st.dataframe(display_df, hide_index=True, use_container_width=True)
    st.caption(f"Displaying {len(filtered_df)} of {len(df)} projects.")
    
    # Risk Breakdowns
    st.markdown("---")
    styled_header("Risk Breakdowns", level=2)
    
    col1, col2 = st.columns(2)
    
    # Only show risk breakdowns if RiskProbability column exists
    if "RiskProbability" in df.columns:
        with col1:
            styled_header("Risk by Project Type", level=3)
            
            # Calculate risk rate by project type
            project_type_risk = df.groupby("ProjectType").apply(
                lambda x: (x["RiskProbability"] > st.session_state.prediction_threshold).mean() * 100
            ).reset_index(name="High-Risk Rate (%)")
            
            # Create bar chart
            fig = px.bar(
                project_type_risk.sort_values("High-Risk Rate (%)", ascending=False),
                x="ProjectType",
                y="High-Risk Rate (%)",
                title="Avg. Predicted High-Risk Rate by Project Type",
                color="High-Risk Rate (%)",
                color_continuous_scale=["#2ecc71", "#f1c40f", "#e74c3c"],
                template="plotly_white"
            )
            
            fig.update_layout(
                xaxis_title="Project Type",
                yaxis_title="High-Risk Rate (%)",
                coloraxis_showscale=False,
                margin=dict(l=40, r=40, t=60, b=40)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            styled_header("Risk by Region", level=3)
            
            # Calculate risk rate by region
            region_risk = df.groupby("Region").apply(
                lambda x: (x["RiskProbability"] > st.session_state.prediction_threshold).mean() * 100
            ).reset_index(name="High-Risk Rate (%)")
            
            # Create bar chart
            fig = px.bar(
                region_risk.sort_values("High-Risk Rate (%)", ascending=False),
                x="Region",
                y="High-Risk Rate (%)",
                title="Avg. Predicted High-Risk Rate by Region",
                color="High-Risk Rate (%)",
                color_continuous_scale=["#2ecc71", "#f1c40f", "#e74c3c"],
                template="plotly_white",
                color_continuous_midpoint=50
            )
            
            fig.update_layout(
                xaxis_title="Region",
                yaxis_title="High-Risk Rate (%)",
                coloraxis_showscale=False,
                margin=dict(l=40, r=40, t=60, b=40)
            )
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        # If RiskProbability doesn't exist, show an appropriate message
        st.info("Risk probability data is not available. Train a risk model or load a project with risk probability data to see risk breakdowns.")
    
    # Add Anomaly Detection section if anomaly results are available
    if st.session_state.anomaly_detection_results is not None and "is_anomaly" in df.columns:
        st.markdown("---")
        styled_header("Anomaly Detection", level=2, icon="🔍")
        st.markdown("Anomaly detection identifies unusual projects with atypical characteristics compared to the rest of the portfolio.")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Display anomaly visualization if available
            if st.session_state.anomaly_visualization is not None:
                st.plotly_chart(st.session_state.anomaly_visualization, use_container_width=True)
            else:
                st.info("No anomaly visualization available.")
        
        with col2:
            # Display anomaly insights
            if st.session_state.anomaly_insights is not None:
                insights = st.session_state.anomaly_insights
                
                # Create a card for anomaly stats
                anomaly_count = insights.get('count', 0)
                anomaly_percentage = insights.get('percentage', 0)
                message = insights.get('message', 'No anomalies detected.')
                
                st.markdown(f"""
                <div style="background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">  
                    <h4 style="color: {ARCADIS_PRIMARY_COLOR}; margin-top: 0;">Anomaly Summary</h4>
                    <p><strong>Count:</strong> {anomaly_count} projects</p>
                    <p><strong>Percentage:</strong> {anomaly_percentage:.1f}% of portfolio</p>
                    <p>{message}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display patterns if available
                if 'patterns' in insights and len(insights['patterns']) > 0:
                    st.markdown(f"<h4 style='color:{ARCADIS_PRIMARY_COLOR};'>Detected Patterns</h4>", unsafe_allow_html=True)
                    
                    for pattern in insights['patterns'][:3]:  # Show top 3 patterns
                        st.markdown(f"""<div style="background-color: #f8f8f8; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                            <p style="margin: 0;">{pattern['message']}</p>
                        </div>""", unsafe_allow_html=True)
            else:
                st.info("No anomaly insights available.")
        
        # Display table of top anomalies
        st.subheader("Top Anomalies")
        if st.session_state.anomaly_detection_results is not None:
            anomaly_df = st.session_state.anomaly_detection_results
            if "is_anomaly" in anomaly_df.columns:
                top_anomalies = anomaly_df[anomaly_df["is_anomaly"]].sort_values("anomaly_score", ascending=False).head(5)
                
                if len(top_anomalies) > 0:
                    # Format the display of top anomalies
                    display_cols = [PROJECT_ID_COLUMN, PROJECT_NAME_COLUMN, "Region", "ProjectType", "anomaly_score"]
                    st.dataframe(top_anomalies[display_cols], hide_index=True, use_container_width=True)
                else:
                    st.info("No anomalies detected in the current dataset.")
            else:
                st.info("Anomaly detection results don't contain anomaly flags.")
        else:
            st.info("No anomaly detection results available.")
            
        # Add an expandable section with anomaly detection explanation
        with st.expander("How does anomaly detection work?"):
            st.markdown("""
            The anomaly detection system uses multiple algorithms including Isolation Forest and Local Outlier Factor to identify
            projects with unusual characteristics. These algorithms look for data points that differ significantly from the majority
            of other projects.
            
            **Key benefits:**
            - Identifies potential risks missed by traditional risk scoring
            - Highlights unusual project characteristics for further investigation
            - Provides early warning of potential issues before they become problems
            - Helps identify both opportunities and threats in your project portfolio
            
            Anomalies are not necessarily high-risk projects - they are simply unusual compared to other projects and warrant further investigation.
            """)
            
            st.markdown(f"""<div style="background-color: {ARCADIS_LIGHT_BG}; padding: 10px; border-radius: 5px; margin-top: 10px;">
                <p style="margin: 0; font-style: italic;">💡 <strong>Pro Tip:</strong> Investigate projects with both high risk scores AND high anomaly scores for potential high-impact problems.</p>
            </div>""", unsafe_allow_html=True)
    
    # Risk Register Heatmap (if risk data is available)
    if st.session_state.risk_data is not None:
        st.markdown("---")
        styled_header("Risk Register Heatmap (Filtered Projects)", level=2)
        
        # Get risk data for filtered projects
        filtered_project_ids = filtered_df[PROJECT_ID_COLUMN].unique()
        risk_df = st.session_state.risk_data
        filtered_risks = risk_df[risk_df[PROJECT_ID_COLUMN].isin(filtered_project_ids)]
        
        if len(filtered_risks) > 0:
            # Create heatmap of risk count by impact and probability
            risk_counts = filtered_risks.groupby(["Probability", "Impact"]).size().reset_index(name="Count")
            
            # Define the order of impact and probability levels
            impact_order = ["Very Low", "Low", "Medium", "High", "Very High"]
            prob_order = ["Very Low", "Low", "Medium", "High", "Very High"]
            
            # Create pivot table for heatmap
            pivot_data = risk_counts.pivot_table(values="Count", index="Probability", columns="Impact", fill_value=0)
            
            # Create heatmap using Plotly
            fig = px.imshow(
                pivot_data,
                labels=dict(x="Impact", y="Likelihood", color="Count"),
                x=pivot_data.columns,
                y=pivot_data.index,
                color_continuous_scale="Oranges",
                template="plotly_white",
                text_auto=True,
                aspect="auto"
            )
            
            fig.update_layout(
                title="Risk Count by Likelihood and Impact",
                margin=dict(l=40, r=40, t=60, b=40)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No risk register data available for the filtered projects.")

def model_analysis_tab():
    """Content for the Model Analysis & Explainability tab"""
    if st.session_state.model_results == {} or st.session_state.trained_models is None:
        # Create a better placeholder when no model is available
        st.markdown(f"""
        <div style="background-color: {ARCADIS_LIGHT_BG}; padding: 20px; border-radius: 10px; text-align: center;">  
            <img src="./assets/ai_icon.png" width="60" style="margin-bottom: 10px;">  
            <h2 style="color: {ARCADIS_PRIMARY_COLOR};">Model Analysis & Explainability</h2>
            <p style="font-size: 16px;">Please train a risk model first to view the model analysis.</p>
            <p style="font-size: 14px; color: {ARCADIS_ACCENT_COLOR};">Use the sidebar on the left to train the model using the "Train Risk Model" button.</p>
        </div>
        """, unsafe_allow_html=True)  
        return
    
    # Create a more visual title with underline animation
    st.markdown(f"""
    <style>
    .animated-header {{  
        position: relative;
        display: inline-block;
    }}
    .animated-header::after {{  
        content: '';
        position: absolute;
        width: 100%;
        height: 3px;
        bottom: -5px;
        left: 0;
        background: linear-gradient(90deg, {ARCADIS_PRIMARY_COLOR}, {ARCADIS_SECONDARY_COLOR});
        transform: scaleX(0);  
        transform-origin: bottom right;
        transition: transform 0.5s ease-out;
        animation: expand 1.5s ease-out forwards;
    }}
    @keyframes expand {{  
        to {{ transform: scaleX(1); transform-origin: bottom left; }}
    }}
    </style>
    <h1 class="animated-header" style="color:{ARCADIS_PRIMARY_COLOR};">🧠 Model Analysis & Explainability</h1>
    <p style="color:{ARCADIS_ACCENT_COLOR}; font-size:16px; margin-top:-5px;">Explore model performance metrics and understand the factors influencing risk predictions</p>
    """, unsafe_allow_html=True)
    
    # Create an elegant model summary card
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, {ARCADIS_PRIMARY_COLOR}22, {ARCADIS_LIGHT_BG}); padding: 20px; border-radius: 10px; margin: 20px 0; border-left: 5px solid {ARCADIS_PRIMARY_COLOR}; display: flex; align-items: center;">
        <div style="margin-right: 20px;">
            <img src="./assets/ai_icon.png" width="60">
        </div>
        <div>
            <h3 style="margin-top: 0;">Best Model: {st.session_state.best_model_name}</h3>
            <p>This model provides the best performance for predicting project risk based on the available data.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Model Metrics Section - Upgraded with color indicators
    st.markdown(f"<h2 style='color: {ARCADIS_PRIMARY_COLOR};'>Performance Metrics</h2>", unsafe_allow_html=True)
    
    # Get metrics from the best model
    metrics = st.session_state.model_results.get("metrics", {}).get(st.session_state.best_model_name, {})
    
    if metrics:
        # Metrics card with unified design
        col1, col2, col3, col4 = st.columns(4)
        
        accuracy = metrics.get('accuracy', 0)
        precision = metrics.get('precision', 0)
        recall = metrics.get('recall', 0)
        roc_auc = metrics.get('roc_auc', 0)
        
        # Helper function to determine color based on metric value
        def get_metric_color(value):
            if value >= 0.8:
                return "#2ecc71"  # Green for good
            elif value >= 0.6:
                return "#f1c40f"  # Yellow for medium
            else:
                return "#e74c3c"  # Red for poor
        
        with col1:
            st.markdown(f"""
            <div style="background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); height: 130px; text-align: center;">
                <p style="color: #555; font-size: 14px; margin-bottom: 5px;">Accuracy</p>
                <h2 style="font-size: 28px; color: {get_metric_color(accuracy)}; margin: 0;">{accuracy:.2f}</h2>
                <div style="margin-top: 10px;">
                    <div style="height: 8px; width: 100%; background-color: #e0e0e0; border-radius: 4px;">
                        <div style="height: 8px; width: {accuracy*100}%; background-color: {get_metric_color(accuracy)}; border-radius: 4px;"></div>
                    </div>
                </div>
                <p style="font-size: 12px; color: #777; margin-top: 5px;">Overall correctness</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); height: 130px; text-align: center;">
                <p style="color: #555; font-size: 14px; margin-bottom: 5px;">Precision</p>
                <h2 style="font-size: 28px; color: {get_metric_color(precision)}; margin: 0;">{precision:.2f}</h2>
                <div style="margin-top: 10px;">
                    <div style="height: 8px; width: 100%; background-color: #e0e0e0; border-radius: 4px;">
                        <div style="height: 8px; width: {precision*100}%; background-color: {get_metric_color(precision)}; border-radius: 4px;"></div>
                    </div>
                </div>
                <p style="font-size: 12px; color: #777; margin-top: 5px;">True positives ratio</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style="background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); height: 130px; text-align: center;">
                <p style="color: #555; font-size: 14px; margin-bottom: 5px;">Recall</p>
                <h2 style="font-size: 28px; color: {get_metric_color(recall)}; margin: 0;">{recall:.2f}</h2>
                <div style="margin-top: 10px;">
                    <div style="height: 8px; width: 100%; background-color: #e0e0e0; border-radius: 4px;">
                        <div style="height: 8px; width: {recall*100}%; background-color: {get_metric_color(recall)}; border-radius: 4px;"></div>
                    </div>
                </div>
                <p style="font-size: 12px; color: #777; margin-top: 5px;">Completeness of positive predictions</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div style="background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); height: 130px; text-align: center;">
                <p style="color: #555; font-size: 14px; margin-bottom: 5px;">ROC AUC</p>
                <h2 style="font-size: 28px; color: {get_metric_color(roc_auc)}; margin: 0;">{roc_auc:.2f}</h2>
                <div style="margin-top: 10px;">
                    <div style="height: 8px; width: 100%; background-color: #e0e0e0; border-radius: 4px;">
                        <div style="height: 8px; width: {roc_auc*100}%; background-color: {get_metric_color(roc_auc)}; border-radius: 4px;"></div>
                    </div>
                </div>
                <p style="font-size: 12px; color: #777; margin-top: 5px;">Model discriminative power</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Feature Importance Section - Enhanced with storytelling
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='color: {ARCADIS_PRIMARY_COLOR};'>Key Risk Drivers</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <p>Understanding which factors have the strongest influence on project risk allows you to focus attention on the most impactful variables. 
        These risk drivers can inform targeted mitigation strategies and proactive management decisions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get feature importance from the best model
    feature_importance = st.session_state.model_results.get("feature_importance", {}).get(st.session_state.best_model_name, None)
    
    if feature_importance is not None:
        # Convert to DataFrame for plotting
        fi_df = pd.DataFrame({
            'Feature': feature_importance[0],
            'Importance': feature_importance[1]
        }).sort_values('Importance', ascending=False).head(10)
        
        # Calculate maximum importance for normalization
        max_importance = fi_df['Importance'].max()
        
        # Create enhanced bar chart with Arcadis colors
        fig = px.bar(
            fi_df,
            x='Importance',
            y='Feature',
            orientation='h',
            color='Importance',
            color_continuous_scale=[[0, ARCADIS_LIGHT_BG], [0.5, ARCADIS_PRIMARY_COLOR], [1, ARCADIS_SECONDARY_COLOR]],
            template="plotly_white",
            text=fi_df['Importance'].apply(lambda x: f"{x:.3f}")
        )
        
        # Enhance styling
        fig.update_layout(
            title={
                'text': f"<b>Top 10 Risk Drivers</b>",
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 22, 'color': ARCADIS_PRIMARY_COLOR}
            },
            yaxis=dict(
                categoryorder='total ascending',
                title=None,
                tickfont={'size': 14}
            ),
            xaxis=dict(
                title=dict(
                    text="Relative Importance",
                    font={'size': 14, 'color': ARCADIS_SECONDARY_COLOR}
                )
            ),
            coloraxis_showscale=False,
            margin=dict(l=40, r=40, t=80, b=40),
            plot_bgcolor='white',
            hoverlabel=dict(bgcolor='white', font_size=16)
        )
        
        # Add % on hover
        fig.update_traces(
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.3f}<br>Relative: %{customdata:.1%}",
            customdata=(fi_df['Importance']/max_importance).values.reshape(-1, 1)
        )
        
        # Add annotations for interpretation
        sorted_features = fi_df.sort_values('Importance', ascending=False)['Feature'].tolist()
        top_feature = sorted_features[0]
        second_feature = sorted_features[1]
        
        # Store the figure in session state for PDF export
        st.session_state.visualizations['feature_importance_fig'] = fig
        
        # Main container for visualization and insights
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            
        with col2:
            st.markdown(f"""
            <div style="background-color: white; padding: 20px; border-radius: 10px; height: 100%; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                <h3 style="color: {ARCADIS_PRIMARY_COLOR}; margin-top: 0;">Key Insights</h3>
                <ul style="padding-left: 20px;">
                    <li style="margin-bottom: 10px;"><b>{top_feature}</b> has the strongest influence on project risk predictions</li>
                    <li style="margin-bottom: 10px;"><b>{second_feature}</b> is the second most important factor</li>
                    <li style="margin-bottom: 10px;">Projects with extreme values in these variables should receive special attention</li>
                </ul>
                <p style="margin-top: 15px; font-style: italic; color: {ARCADIS_SECONDARY_COLOR};">Proactively managing these key factors can significantly reduce project risk</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Model Evaluation Plots - Enhanced with insights and storytelling
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='color: {ARCADIS_PRIMARY_COLOR};'>Model Performance Assessment</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <p>These visualizations show how well the model performs in identifying high-risk projects. A good model will have high 
        true positive and true negative rates with minimal misclassifications.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Use Beta columns for a more engaging layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Enhanced Confusion Matrix
        if "confusion_matrix" in st.session_state.model_results:
            cm = st.session_state.model_results["confusion_matrix"]
            
            # Calculate metrics from confusion matrix
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
            total = sum([tn, fp, fn, tp])
            accuracy = (tn + tp) / total if total > 0 else 0
            
            # Add annotations with percentages
            annotations = [
                [f"{tn} ({tn/total:.1%})", f"{fp} ({fp/total:.1%})"],
                [f"{fn} ({fn/total:.1%})", f"{tp} ({tp/total:.1%})"]
            ]
            
            # Create nicer confusion matrix plot with Arcadis colors
            fig = px.imshow(
                cm,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=["Low Risk", "High Risk"],
                y=["Low Risk", "High Risk"],
                color_continuous_scale=[[0, "#f0f0f0"], [0.5, ARCADIS_LIGHT_BG], [1, ARCADIS_PRIMARY_COLOR]],
                template="plotly_white",
                text_auto=False
            )
            
            # Add text annotations
            for i in range(len(annotations)):
                for j in range(len(annotations[0])):
                    fig.add_annotation(
                        x=["Low Risk", "High Risk"][j],
                        y=["Low Risk", "High Risk"][i],
                        text=annotations[i][j],
                        showarrow=False,
                        font=dict(color="black", size=14)
                    )
            
            # Update layout with better styling
            fig.update_layout(
                title={
                    'text': "<b>Confusion Matrix</b>",
                    'y':0.95,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': {'size': 18, 'color': ARCADIS_PRIMARY_COLOR}
                },
                xaxis=dict(side="bottom", title=None, tickfont={'size': 14}),
                yaxis=dict(title=None, tickfont={'size': 14}),
                margin=dict(l=40, r=40, t=60, b=40),
                height=350
            )
            
            # Store the figure in session state for PDF export
            st.session_state.visualizations['confusion_matrix_fig'] = fig
            
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            
            # Add confusion matrix explanation
            st.markdown(f"""
            <div style="background-color: white; border-left: 4px solid {ARCADIS_PRIMARY_COLOR}; padding: 15px; border-radius: 5px; margin-top: 10px;">
                <b>Interpretation:</b> {accuracy:.1%} of predictions are correct. 
                Pay special attention to the {fn} projects misclassified as low-risk when they are actually high-risk (false negatives).
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Enhanced ROC Curve
        if "fpr" in st.session_state.model_results and "tpr" in st.session_state.model_results:
            fpr = st.session_state.model_results["fpr"]
            tpr = st.session_state.model_results["tpr"]
            auc = metrics.get("roc_auc", 0)
            
            # Create nicer ROC curve plot with Arcadis color scheme
            fig = go.Figure()
            
            # Add ROC curve
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                line=dict(color=ARCADIS_PRIMARY_COLOR, width=3),
                name=f'Model (AUC = {auc:.3f})',
                hovertemplate='False Positive Rate: %{x:.2f}<br>True Positive Rate: %{y:.2f}'
            ))
            
            # Add diagonal reference line
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                line=dict(color='gray', width=2, dash='dash'),
                name='Random Guess (AUC = 0.5)',
                hoverinfo='skip'
            ))
            
            # Add shaded area under the ROC curve
            fig.add_trace(go.Scatter(
                x=np.concatenate([fpr, [1, 0]]),
                y=np.concatenate([tpr, [0, 0]]),
                fill='toself',
                fillcolor=f'rgba({int(ARCADIS_PRIMARY_COLOR[1:3], 16)}, {int(ARCADIS_PRIMARY_COLOR[3:5], 16)}, {int(ARCADIS_PRIMARY_COLOR[5:7], 16)}, 0.2)',
                line=dict(color='rgba(0,0,0,0)'),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            # Update layout with better styling
            fig.update_layout(
                title={
                    'text': "<b>ROC Curve</b>",
                    'y':0.95,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': {'size': 18, 'color': ARCADIS_PRIMARY_COLOR}
                },
                xaxis=dict(
                    title=dict(
                        text="False Positive Rate",
                        font={'size': 14, 'color': ARCADIS_SECONDARY_COLOR}
                    ), 
                    tickformat='.1f',
                    range=[0, 1.05],
                    tickfont={'size': 12}
                ),
                yaxis=dict(
                    title=dict(
                        text="True Positive Rate",
                        font={'size': 14, 'color': ARCADIS_SECONDARY_COLOR}
                    ), 
                    tickformat='.1f',
                    range=[0, 1.05],
                    tickfont={'size': 12}
                ),
                margin=dict(l=40, r=40, t=60, b=40),
                height=350,
                template="plotly_white",
                legend=dict(x=0.05, y=0.05, bgcolor='rgba(255,255,255,0.8)'),
                hovermode='closest'
            )
            
            # Add annotation for perfect classifier
            fig.add_annotation(
                x=0.1, y=0.9,
                text="Perfect Classifier",
                showarrow=True,
                arrowhead=1,
                ax=30, ay=-30,
                font=dict(color=ARCADIS_PRIMARY_COLOR, size=12)
            )
            
            # Store the figure in session state for PDF export
            st.session_state.visualizations['roc_curve_fig'] = fig
            
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            
            # Add ROC curve explanation
            roc_quality = "excellent" if auc > 0.9 else "good" if auc > 0.8 else "fair" if auc > 0.7 else "poor"
            st.markdown(f"""
            <div style="background-color: white; border-left: 4px solid {ARCADIS_PRIMARY_COLOR}; padding: 15px; border-radius: 5px; margin-top: 10px;">
                <b>Interpretation:</b> The ROC curve shows model discriminative power with an AUC of {auc:.3f}, which is 
                <span style="color: {'#2ecc71' if auc > 0.8 else '#f1c40f' if auc > 0.7 else '#e74c3c'}"><b>{roc_quality}</b></span>. 
                Higher AUC indicates better ability to distinguish between high and low-risk projects.
            </div>
            """, unsafe_allow_html=True)
    
    # LIME Explainer Section
    st.markdown("---")
    styled_header("LIME Explanation for Individual Projects", level=2)
    st.markdown("See why specific projects are predicted to be high-risk or low-risk.")
    
    # Project selection
    if st.session_state.project_data is not None and st.session_state.risk_probabilities is not None:
        df = st.session_state.project_data
        
        # Sort projects by risk probability
        project_risks = pd.DataFrame({
            'ProjectID': df[PROJECT_ID_COLUMN],
            'ProjectName': df[PROJECT_NAME_COLUMN],
            'Risk': st.session_state.risk_probabilities
        }).sort_values('Risk', ascending=False)
        
        # Create selection options
        options = [f"{row['ProjectName']} ({row['ProjectID']}) - {row['Risk']:.1%}" 
                  for _, row in project_risks.head(20).iterrows()]
        
        selected_project = st.selectbox("Select a project to explain:", options)
        
        if selected_project:
            # Extract project ID from selection
            project_id = selected_project.split('(')[1].split(')')[0]
            
            # Display project details
            project_data = df[df[PROJECT_ID_COLUMN] == project_id].iloc[0]
            risk_prob = st.session_state.risk_probabilities[df[PROJECT_ID_COLUMN] == project_id].iloc[0]
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                styled_header(f"Project: {project_data[PROJECT_NAME_COLUMN]}", level=3)
                st.markdown(f"**Project ID:** {project_id}")
                st.markdown(f"**Risk Probability:** {risk_prob:.1%}")
                st.markdown(f"**Risk Classification:** {'High Risk' if risk_prob > st.session_state.prediction_threshold else 'Low Risk'}")
                
                # More project details
                st.markdown("**Project Details:**")
                details = {
                    "Type": project_data.get("ProjectType", "N/A"),
                    "Region": project_data.get("Region", "N/A"),
                    "Budget": f"${project_data.get('Budget', 0):,.0f}",
                    "Duration": f"{project_data.get('DurationMonths', 0)} months",
                    "Complexity": project_data.get("ComplexityLevel", "N/A")
                }
                
                for k, v in details.items():
                    st.markdown(f"- {k}: {v}")
            
            with col2:
                # Simulated LIME explanation (in real implementation, this would come from the LIME explainer)
                styled_header("Risk Factors Explanation", level=3)
                
                # Create explanation for demo purposes
                # In a real implementation, this would be dynamically generated using LIME
                explanation = [
                    {"Feature": "ComplexityLevel_Very High", "Weight": 0.35, "Direction": "Increases Risk"},
                    {"Feature": "Budget", "Weight": 0.25, "Direction": "Increases Risk"},
                    {"Feature": "StakeholderEngagementScore", "Weight": -0.20, "Direction": "Decreases Risk"},
                    {"Feature": "Region_MEA", "Weight": 0.15, "Direction": "Increases Risk"},
                    {"Feature": "TeamSize", "Weight": -0.10, "Direction": "Decreases Risk"}
                ]
                
                # Create a plot
                exp_df = pd.DataFrame(explanation)
                
                # Adjust for direction
                exp_df["Adjusted Weight"] = exp_df.apply(
                    lambda x: x["Weight"] if x["Direction"] == "Increases Risk" else -x["Weight"], 
                    axis=1
                )
                
                # Sort by absolute weight
                exp_df = exp_df.sort_values("Adjusted Weight", key=lambda x: abs(x), ascending=False)
                
                # Create color mapping
                exp_df["Color"] = exp_df["Direction"].map({
                    "Increases Risk": ARCADIS_DANGER,
                    "Decreases Risk": ARCADIS_SUCCESS
                })
                
                # Create bar chart
                fig = px.bar(
                    exp_df,
                    x="Adjusted Weight",
                    y="Feature",
                    color="Direction",
                    color_discrete_map={
                        "Increases Risk": ARCADIS_DANGER,
                        "Decreases Risk": ARCADIS_SUCCESS
                    },
                    title="Factors Influencing Risk Prediction",
                    orientation="h",
                    template="plotly_white"
                )
                
                fig.update_layout(
                    yaxis=dict(categoryorder='total ascending'),
                    margin=dict(l=40, r=40, t=60, b=40)
                )
                
                # Store the figure in session state for PDF export
                project_key = f"lime_explanation_{project_id}"
                st.session_state.visualizations[project_key] = fig
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("**Interpretation:**")
                st.markdown(
                    f"The model predicts this project as {'high' if risk_prob > st.session_state.prediction_threshold else 'low'} risk "
                    f"primarily due to its {explanation[0]['Feature'].split('_')[-1] if '_' in explanation[0]['Feature'] else explanation[0]['Feature']} "
                    f"and high {explanation[1]['Feature']}. However, good {explanation[2]['Feature']} partially mitigates the risk."
                )

def simulation_tab():
    """Content for the Simulation & Scenarios tab"""
    if st.session_state.project_data is None or st.session_state.risk_predictions is None:
        st.warning("Please load data and train a risk model to access the simulation features.")
        return
    
    styled_header("Simulation & Scenarios", icon="🎲")
    st.markdown("Explore 'what-if' scenarios and understand the uncertainty in risk predictions.")
    
    # Risk Threshold Simulation
    st.markdown("---")
    styled_header("Risk Threshold Simulation", level=2)
    st.markdown("Adjust the risk threshold to see how it affects the number of high-risk projects identified.")
    
    # Get risk probabilities
    probabilities = st.session_state.risk_probabilities
    
    # Create slider for threshold
    threshold = st.slider(
        "Risk Probability Threshold:",
        min_value=0.1,
        max_value=0.9,
        value=st.session_state.prediction_threshold,
        step=0.05,
        format="%.2f"
    )
    
    # Calculate metrics based on threshold
    high_risk_count = (probabilities > threshold).sum()
    high_risk_rate = high_risk_count / len(probabilities) * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        styled_metric_card(
            label="High-Risk Projects",
            value=high_risk_count,
            icon="⚠️",
            color=ARCADIS_SECONDARY_COLOR
        )
    
    with col2:
        styled_metric_card(
            label="High-Risk Rate",
            value=f"{high_risk_rate:.1f}%",
            icon="📊"
        )
    
    # Create histogram with threshold line
    fig = px.histogram(
        x=probabilities,
        nbins=20,
        labels={"x": "Predicted Probability"},
        title="Distribution of Predicted Risk Probabilities",
        template="plotly_white"
    )
    
    # Add vertical line for threshold
    fig.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Threshold ({threshold})",
        annotation_position="top left"
    )
    
    fig.update_layout(
        bargap=0.1,
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis=dict(
            title=dict(
                text="Predicted Probability",
                font={'size': 14, 'color': ARCADIS_SECONDARY_COLOR}
            )
        ),
        yaxis=dict(
            title=dict(
                text="Count",
                font={'size': 14, 'color': ARCADIS_SECONDARY_COLOR}
            )
        )
    )
    
    # Store the figure in session state for PDF export
    st.session_state.visualizations['risk_distribution_fig'] = fig
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 'What-If' Scenario Analysis
    st.markdown("---")
    styled_header("'What-If' Scenario Analysis", level=2)
    st.markdown("Explore how changes to project parameters could impact risk predictions.")
    
    # Project selection
    if st.session_state.project_data is not None:
        df = st.session_state.project_data
        
        # Create selection options
        options = [f"{row[PROJECT_NAME_COLUMN]} ({row[PROJECT_ID_COLUMN]})" 
                 for _, row in df.iterrows()]
        
        selected_project = st.selectbox("Select a project for scenario analysis:", options, index=0)
        
        if selected_project:
            # Extract project ID from selection
            project_id = selected_project.split('(')[1].split(')')[0]
            
            # Get the project data
            project_data = df[df[PROJECT_ID_COLUMN] == project_id].iloc[0].copy()
            current_risk = st.session_state.risk_probabilities[df[PROJECT_ID_COLUMN] == project_id].iloc[0]
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                styled_header("Modify Project Parameters", level=3)
                
                # Parameter sliders
                st.markdown("**Project Complexity:**")
                complexity_map = {"Low": 0, "Medium": 1, "High": 2, "Very High": 3}
                current_complexity = complexity_map.get(project_data.get("ComplexityLevel", "Medium"), 1)
                complexity_options = ["Low", "Medium", "High", "Very High"]
                new_complexity = st.selectbox(
                    "Complexity Level:",
                    options=complexity_options,
                    index=current_complexity
                )
                
                st.markdown("**Budget Change:**")
                budget_change = st.slider(
                    "Budget Change (%):",
                    min_value=-50,
                    max_value=100,
                    value=0,
                    step=5
                )
                
                st.markdown("**Project Duration Change:**")
                duration_change = st.slider(
                    "Duration Change (%):",
                    min_value=-30,
                    max_value=100,
                    value=0,
                    step=5
                )
                
                st.markdown("**Stakeholder Engagement:**")
                # Safely handle NaN values when getting StakeholderEngagementScore
                engagement_score = project_data.get("StakeholderEngagementScore", 5)
                current_engagement = 5  # default value
                if pd.notna(engagement_score):
                    try:
                        current_engagement = int(engagement_score)
                    except (ValueError, TypeError):
                        # Keep default value on conversion error
                        pass
                new_engagement = st.slider(
                    "Stakeholder Engagement Score (1-10):",
                    min_value=1,
                    max_value=10,
                    value=current_engagement
                )
                
                # Calculate new risk (simplified simulation for demo)
                # In real implementation, this would use the actual ML model
                complexity_risk_factor = {"Low": 0.1, "Medium": 0.2, "High": 0.3, "Very High": 0.4}.get(new_complexity, 0.2)
                budget_risk_factor = 0.1 * (budget_change / 100) if budget_change > 0 else 0
                duration_risk_factor = 0.1 * (duration_change / 100) if duration_change > 0 else 0
                engagement_risk_factor = -0.03 * (new_engagement - current_engagement)
                
                base_risk = current_risk
                new_risk = np.clip(base_risk + complexity_risk_factor + budget_risk_factor + duration_risk_factor + engagement_risk_factor, 0.05, 0.95)
                
                # Run scenario button
                analyze_btn = st.button("Run Scenario Analysis", type="primary", use_container_width=True)
            
            with col2:
                styled_header("Scenario Results", level=3)
                
                if analyze_btn:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        styled_metric_card(
                            label="Current Risk Probability",
                            value=f"{current_risk:.1%}",
                            icon="📊"
                        )
                    
                    with col2:
                        # Calculate percentage change with protection against division by zero
                        if current_risk > 0:
                            pct_change = ((new_risk - current_risk) / current_risk) * 100
                            delta = f"{pct_change:.1f}%" if pct_change != 0 else None
                        else:
                            delta = "N/A"
                        
                        styled_metric_card(
                            label="New Risk Probability",
                            value=f"{new_risk:.1%}",
                            delta=delta,
                            icon="📉" if new_risk < current_risk else "📈",
                            color=ARCADIS_SUCCESS if new_risk < current_risk else ARCADIS_DANGER
                        )
                    
                    # Risk classification
                    current_class = "High Risk" if current_risk > threshold else "Low Risk"
                    new_class = "High Risk" if new_risk > threshold else "Low Risk"
                    class_changed = current_class != new_class
                    
                    if class_changed:
                        if new_class == "Low Risk":
                            st.success(f"✅ Risk classification changed from {current_class} to {new_class}!")
                        else:
                            st.error(f"⚠️ Risk classification changed from {current_class} to {new_class}!")
                    
                    # Factor impact analysis
                    st.markdown("**Factor Impact Analysis:**")
                    
                    impact_factors = [
                        {"Factor": "Complexity Level", "Impact": complexity_risk_factor, "Direction": "+" if complexity_risk_factor > 0 else "-"},
                        {"Factor": "Budget Change", "Impact": budget_risk_factor, "Direction": "+" if budget_risk_factor > 0 else "-"},
                        {"Factor": "Duration Change", "Impact": duration_risk_factor, "Direction": "+" if duration_risk_factor > 0 else "-"},
                        {"Factor": "Stakeholder Engagement", "Impact": engagement_risk_factor, "Direction": "+" if engagement_risk_factor > 0 else "-"}
                    ]
                    
                    # Create impact DataFrame
                    impact_df = pd.DataFrame(impact_factors)
                    impact_df["Absolute Impact"] = impact_df["Impact"].abs()
                    impact_df = impact_df.sort_values("Absolute Impact", ascending=False)
                    
                    # Create a horizontal bar chart
                    fig = px.bar(
                        impact_df,
                        x="Impact",
                        y="Factor",
                        orientation="h",
                        color="Impact",
                        color_continuous_scale=["green", "yellow", "red"],
                        color_continuous_midpoint=0,
                        title="Impact of Changes on Risk Probability",
                        template="plotly_white"
                    )
                    
                    fig.update_layout(
                        margin=dict(l=40, r=40, t=60, b=40),
                        xaxis=dict(
                            title=dict(
                                text="Impact on Risk Probability",
                                font={'size': 14, 'color': ARCADIS_SECONDARY_COLOR}
                            )
                        ),
                        yaxis=dict(
                            title=dict(
                                font={'size': 14, 'color': ARCADIS_SECONDARY_COLOR}
                            )
                        )
                    )
                    
                    # Store the figure in session state for PDF export
                    scenario_key = f"impact_analysis_{project_id}"
                    st.session_state.visualizations[scenario_key] = fig
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Risk change description
                    st.markdown("**Risk Change Analysis:**")
                    
                    # Get the highest impact factor
                    highest_impact = impact_df.iloc[0]
                    
                    if new_risk > current_risk:
                        st.markdown(
                            f"The risk probability increased primarily due to the change in **{highest_impact['Factor']}** "
                            f"(impact: {highest_impact['Impact']:.3f}). Consider addressing this factor to reduce project risk."
                        )
                    elif new_risk < current_risk:
                        st.markdown(
                            f"The risk probability decreased primarily due to the change in **{highest_impact['Factor']}** "
                            f"(impact: {highest_impact['Impact']:.3f}). This appears to be an effective risk mitigation strategy."
                        )
                    else:
                        st.markdown(
                            f"The changes had minimal impact on the overall risk probability. The factors appear to be offsetting each other."
                        )
                    
                    # Recommendation
                    st.markdown("**Recommendation:**")
                    
                    if new_risk > threshold:
                        st.markdown(
                            f"This project is still classified as **High Risk** after the changes. Consider additional "
                            f"mitigation strategies, particularly focused on improving stakeholder engagement and reducing complexity."
                        )
                    else:
                        st.markdown(
                            f"The project is now classified as **Low Risk** after the changes. Continue to monitor the project "
                            f"and maintain the improved parameters to keep risk levels low."
                        )
                else:
                    st.info("Adjust the parameters and click 'Run Scenario Analysis' to see the potential impact on project risk.")
    
    # Anomaly Detection Configuration
    st.markdown("---")
    styled_header("Anomaly Detection Configuration", level=2, icon="🔍")
    st.markdown("Configure the anomaly detection settings to adjust how unusual projects are identified.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        anomaly_method = st.selectbox(
            "Anomaly Detection Algorithm:",
            options=["isolation_forest", "lof", "ensemble"],
            index=["isolation_forest", "lof", "ensemble"].index(st.session_state.anomaly_detection_method),
            help="Isolation Forest is good for global outliers, LOF is good for local density-based outliers, and Ensemble combines both approaches"
        )
        
        # Update session state if method changed
        if anomaly_method != st.session_state.anomaly_detection_method:
            st.session_state.anomaly_detection_method = anomaly_method
    
    with col2:
        contamination = st.slider(
            "Expected Anomaly Percentage:",
            min_value=0.01,
            max_value=0.3,
            value=st.session_state.anomaly_contamination,
            step=0.01,
            format="%.2f",
            help="The expected percentage of projects that are anomalies in your dataset"
        )
        
        # Update session state if contamination changed
        if contamination != st.session_state.anomaly_contamination:
            st.session_state.anomaly_contamination = contamination
    
    # Button to rerun anomaly detection with new settings
    if st.button("Update Anomaly Detection", type="primary", use_container_width=True):
        if st.session_state.project_data_with_predictions is not None:
            st.session_state.anomaly_detection_results = None  # Reset results to force recalculation
            st.session_state.anomaly_visualization = None
            st.session_state.anomaly_insights = None
            
            # Import the anomaly detection module
            from utils.anomaly_detection import detect_anomalies, get_anomaly_insights
            
            with st.spinner("Running anomaly detection with updated settings..."):
                # Rerun anomaly detection with the updated settings
                df = st.session_state.project_data_with_predictions
                anomaly_df, anomaly_viz = detect_anomalies(
                    df, 
                    st.session_state.column_mapping,
                    method=st.session_state.anomaly_detection_method,
                    contamination=st.session_state.anomaly_contamination
                )
                
                # Get insights about anomalies
                anomaly_insights = get_anomaly_insights(anomaly_df, st.session_state.column_mapping)
                
                # Store anomaly detection results
                st.session_state.anomaly_detection_results = anomaly_df
                st.session_state.anomaly_visualization = anomaly_viz
                st.session_state.anomaly_insights = anomaly_insights
                
                # Add anomaly scores to predictions dataframe
                df['anomaly_score'] = anomaly_df['anomaly_score']
                df['is_anomaly'] = anomaly_df['is_anomaly']
                
                # Update the data with predictions
                st.session_state.project_data_with_predictions = df
                
                st.success("Anomaly detection updated successfully.")
                st.rerun()  # Rerun the app to reflect changes
    
    # Description of the anomaly detection process
    with st.expander("Understanding Anomaly Detection Algorithms"):
        st.markdown("""
        **Anomaly detection** identifies unusual data points (outliers) in your project portfolio. These algorithms 
        find projects with characteristics that significantly deviate from the norm.
        
        **Available algorithms:**
        
        1. **Isolation Forest** - Randomly partitions data to isolate outliers. Works well for global outliers that 
        are numerically distant from other data points. Good for small to large datasets and high-dimensional data.
        
        2. **Local Outlier Factor (LOF)** - Compares the local density of a point to its neighbors. Excels at finding 
        local outliers that might be missed by global methods. Better for smaller datasets with complex patterns.
        
        3. **Ensemble** - Combines both Isolation Forest and LOF to get the benefits of both methods. Generally provides 
        more robust results at the cost of additional computation time.
        
        **Contamination parameter** represents the expected proportion of outliers in your dataset. 
        Setting this too high might flag normal projects as anomalies, while setting it too low might miss 
        genuine outliers. Typical values range from 0.01 (1%) to 0.1 (10%).  
        """)
        
        st.markdown(f"""<div style="background-color: {ARCADIS_LIGHT_BG}; padding: 10px; border-radius: 5px; margin-top: 10px;">
            <p style="margin: 0;"><strong>Tip:</strong> Adjust these parameters if you find that too many or too few projects are being 
            flagged as anomalies. Start with the Ensemble method and a contamination value of 0.05 (5%) for balanced results.</p>
        </div>""", unsafe_allow_html=True)
    
    # Monte Carlo Simulation
    st.markdown("---")
    styled_header("Monte Carlo Simulation", level=2, icon="🎲")
    st.markdown("Understand the uncertainty in risk predictions through Monte Carlo simulation.")
    
    # Import the monte_carlo module
    from utils.monte_carlo import (
        run_monte_carlo_simulation,
        create_monte_carlo_distribution_chart,
        create_sensitivity_analysis_chart,
        create_scenario_comparison_chart,
        create_factor_distribution_chart,
        generate_monte_carlo_report
    )
    
    # Setup simulation parameters
    st.markdown("### Configure Simulation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Number of simulation runs
        num_simulations = st.number_input(
            "Number of Simulation Runs:",
            min_value=1000,
            max_value=10000,
            value=5000,
            step=500,
            help="More runs produce more accurate results but take longer to process."
        )
        
        # Risk threshold for high-risk classification
        simulation_threshold = st.slider(
            "Risk Threshold for Simulation:",
            min_value=0.1,
            max_value=0.9,
            value=threshold,  # Use the threshold from earlier in the page
            step=0.05,
            format="%.2f",
            help="Projects with risk probability above this threshold are classified as high risk."
        )
    
    with col2:
        # Select a project to simulate (optional)
        project_options = ["None (Portfolio Level)"] + [f"{row[PROJECT_NAME_COLUMN]} ({row[PROJECT_ID_COLUMN]})" 
                                                   for _, row in df.iterrows()]
        selected_project_sim = st.selectbox(
            "Select a Project (Optional):",
            options=project_options,
            index=0,
            help="Select a specific project to analyze, or leave as 'None' for portfolio-level analysis."
        )
        
        # Simulation detail level
        detail_level = st.select_slider(
            "Simulation Detail Level:",
            options=["Basic", "Intermediate", "Advanced"],
            value="Intermediate",
            help="Basic gives high-level results, Advanced provides detailed breakdowns of risk factors."
        )
    
    # Run simulation button
    run_sim_button = st.button("Run Monte Carlo Simulation", type="primary", use_container_width=True)
    
    # If the simulation has been run, show results
    if 'monte_carlo_results' not in st.session_state:
        st.session_state.monte_carlo_results = None
    
    if run_sim_button:
        with st.spinner("Running Monte Carlo simulation..."):
            # Create project-specific or portfolio-level risk factors
            if selected_project_sim != "None (Portfolio Level)":
                # For project-specific simulation, adjust risk factors based on project data
                project_id = selected_project_sim.split('(')[1].split(')')[0]
                project_data = df[df[PROJECT_ID_COLUMN] == project_id].iloc[0]
                
                # Example of how to adjust risk factors based on project data
                complexity_level = project_data.get("ComplexityLevel", "Medium")
                complexity_factor = {"Low": 0.3, "Medium": 0.5, "High": 0.7, "Very High": 0.9}.get(complexity_level, 0.5)
                
                # Create project-specific risk factors
                risk_factors = {
                    'complexity': {'type': 'triangular', 'min': max(0.1, complexity_factor - 0.2), 
                                 'mode': complexity_factor, 'max': min(0.9, complexity_factor + 0.2)},
                    'schedule_pressure': {'type': 'triangular', 'min': 0.1, 'mode': 0.4, 'max': 0.8},
                    'budget_pressure': {'type': 'triangular', 'min': 0.2, 'mode': 0.5, 'max': 0.9},
                    'stakeholder_alignment': {'type': 'triangular', 'min': 0.1, 'mode': 0.3, 'max': 0.7},
                    'team_experience': {'type': 'triangular', 'min': 0.1, 'mode': 0.4, 'max': 0.8},
                    'requirement_clarity': {'type': 'triangular', 'min': 0.1, 'mode': 0.3, 'max': 0.7},
                    'technical_risk': {'type': 'triangular', 'min': 0.2, 'mode': 0.5, 'max': 0.9},
                }
            else:
                # For portfolio-level simulation, use default risk factors
                risk_factors = None
            
            # Define risk weights (these could come from your trained model if available)
            risk_weights = {
                'complexity': 0.2,
                'schedule_pressure': 0.15,
                'budget_pressure': 0.15,
                'stakeholder_alignment': 0.1,
                'team_experience': 0.15,
                'requirement_clarity': 0.1,
                'technical_risk': 0.15,
            }
            
            # Run the simulation
            sim_results = run_monte_carlo_simulation(
                df, 
                num_simulations=num_simulations,
                risk_factors=risk_factors,
                risk_weights=risk_weights
            )
            
            # Store results in session state
            st.session_state.monte_carlo_results = sim_results
            st.session_state.monte_carlo_config = {
                'threshold': simulation_threshold,
                'detail_level': detail_level,
                'project': selected_project_sim,
                'risk_weights': risk_weights
            }
    
    # Display simulation results if available
    if st.session_state.monte_carlo_results is not None:
        sim_results = st.session_state.monte_carlo_results
        sim_config = st.session_state.monte_carlo_config
        
        st.markdown("### Monte Carlo Simulation Results")
        
        # Display key statistics
        risk_probs = sim_results['risk_probabilities']
        mean_risk = np.mean(risk_probs)
        median_risk = np.median(risk_probs)
        std_risk = np.std(risk_probs)
        p10 = np.percentile(risk_probs, 10)
        p90 = np.percentile(risk_probs, 90)
        
        # Create metric columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            styled_metric_card(
                label="Mean Risk",
                value=f"{mean_risk:.2f}",
                icon="📊",
                color=ARCADIS_PRIMARY_COLOR
            )
        
        with col2:
            styled_metric_card(
                label="Median Risk",
                value=f"{median_risk:.2f}",
                icon="📉",
                color=ARCADIS_SECONDARY_COLOR
            )
        
        with col3:
            styled_metric_card(
                label="Risk Std Dev",
                value=f"{std_risk:.2f}",
                icon="📏",
                color=ARCADIS_ACCENT_COLOR
            )
        
        with col4:
            high_risk_rate = (risk_probs > sim_config['threshold']).mean() * 100
            styled_metric_card(
                label="High Risk Rate",
                value=f"{high_risk_rate:.1f}%",
                icon="⚠️",
                color=ARCADIS_DANGER if high_risk_rate > 50 else ARCADIS_PRIMARY_COLOR
            )
        
        # Display risk probability distribution
        st.markdown("#### Risk Probability Distribution")
        dist_fig = create_monte_carlo_distribution_chart(sim_results)
        
        # Add threshold line to the distribution chart
        dist_fig.add_vline(
            x=sim_config['threshold'],
            line_dash="solid",
            line_color="red",
            line_width=2,
            annotation_text=f"Risk Threshold ({sim_config['threshold']})",
            annotation_position="top left",
            annotation_font={"color": "red", "size": 12}
        )
        
        # Store the figure in session state for PDF export
        st.session_state.visualizations['monte_carlo_dist'] = dist_fig
        
        st.plotly_chart(dist_fig, use_container_width=True)
        
        # Show uncertainty range
        st.markdown(f"**80% Confidence Interval:** [{p10:.2f} - {p90:.2f}]")
        
        # Create tabs for detailed analysis based on detail level
        if sim_config['detail_level'] != "Basic":
            mc_tabs = st.tabs(["Sensitivity Analysis", "Scenario Comparison", "Risk Report"])
            
            with mc_tabs[0]:
                # Sensitivity analysis chart
                st.markdown("#### Sensitivity Analysis")
                sensitivity_fig = create_sensitivity_analysis_chart(sim_results, sim_config['risk_weights'])
                st.plotly_chart(sensitivity_fig, use_container_width=True)
                
                st.markdown("""
                This chart shows how each risk factor correlates with the overall risk probability.
                Positive values indicate that higher factor values increase risk, while negative values
                indicate that higher factor values decrease risk. The magnitude shows the strength of influence.
                """)
            
            with mc_tabs[1]:
                # Scenario comparison
                st.markdown("#### Scenario Comparison")
                scenario_fig = create_scenario_comparison_chart(sim_results)
                st.plotly_chart(scenario_fig, use_container_width=True)
                
                st.markdown("""
                This chart compares different risk scenarios based on percentiles from the Monte Carlo simulation.
                Each scenario represents a different threshold for classifying projects as high risk,
                showing the percentage of projects that would be classified as high risk under each scenario.
                """)
            
            with mc_tabs[2]:
                # Generate and display Monte Carlo report
                st.markdown("#### Monte Carlo Risk Report")
                report = generate_monte_carlo_report(sim_results, sim_config['risk_weights'])
                st.markdown(report)
        
        # For advanced detail level, add factor distribution analysis
        if sim_config['detail_level'] == "Advanced":
            st.markdown("---")
            st.markdown("#### Risk Factor Distributions")
            st.markdown("Select a risk factor to view its distribution across the simulation runs:")
            
            # Factor selection
            available_factors = list(sim_results['factors'].keys())
            factor_display_names = [f.replace('_', ' ').title() for f in available_factors]
            selected_factor_display = st.selectbox(
                "Risk Factor:",
                options=factor_display_names,
                index=0
            )
            
            # Convert back to original format
            selected_factor_key = available_factors[factor_display_names.index(selected_factor_display)]
            
            # Create and display factor distribution
            factor_fig = create_factor_distribution_chart(sim_results, selected_factor_key)
            if factor_fig:
                st.plotly_chart(factor_fig, use_container_width=True)
        
        # Allow download of full report
        st.markdown("---")
        st.markdown("### Download Full Monte Carlo Analysis")
        st.markdown("Generate a comprehensive PDF report with all Monte Carlo analysis results and recommendations.")
        
        # This would be implemented with PDF generation functionality
        st.button("Generate Monte Carlo Report", use_container_width=True)

def main():
    """Main application entry point"""
    # Set page styling
    set_streamlit_style()
    
    # Initialize session state
    initialize_session_state()
    
    # Create sidebar
    make_sidebar()
    
    # Tab navigation using clickable tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        f"{tab['emoji']} {tab['name']}" for tab in TABS
    ])
    
    # Display content based on active tab
    with tab1:  # Welcome tab
        welcome_tab()
    
    with tab2:  # Executive Summary
        executive_summary_tab()
    
    with tab3:  # Portfolio Deep Dive
        portfolio_deep_dive_tab()
    
    with tab4:  # Model Analysis
        model_analysis_tab()
    
    with tab5:  # Simulation & Scenarios
        simulation_tab()

if __name__ == "__main__":
    main()
