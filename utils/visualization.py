"""
Visualization module for RiskLens Pro
Provides functions for creating interactive visualizations
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple, Optional, Union


def plot_feature_importance(feature_names, feature_importance, top_n=10, title="Feature Importance"):
    """
    Create a plotly bar chart of feature importance
    Args:
        feature_names: List of feature names
        feature_importance: Array of feature importance values
        top_n: Number of top features to display
        title: Title for the plot
    Returns:
        go.Figure: Plotly figure object
    """
    # Create a DataFrame for plotting
    if len(feature_names) != len(feature_importance):
        raise ValueError("Length of feature_names and feature_importance must match")
    
    fi_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    })
    
    # Sort by importance
    fi_df = fi_df.sort_values('Importance', ascending=False).head(top_n)
    
    # Create the plot
    fig = px.bar(
        fi_df,
        x='Importance',
        y='Feature',
        title=title,
        orientation='h',
        color='Importance',
        color_continuous_scale=px.colors.sequential.Blues,
        template="plotly_white"
    )
    
    fig.update_layout(
        yaxis=dict(categoryorder='total ascending'),
        coloraxis_showscale=False,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig


def plot_confusion_matrix(cm, class_names=['Low Risk', 'High Risk'], title="Confusion Matrix"):
    """
    Create a plotly heatmap of a confusion matrix
    Args:
        cm: Confusion matrix array (2x2)
        class_names: List of class names
        title: Title for the plot
    Returns:
        go.Figure: Plotly figure object
    """
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=class_names,
        y=class_names,
        color_continuous_scale="Blues",
        template="plotly_white",
        text_auto=True,
        title=title
    )
    
    fig.update_layout(
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig


def plot_roc_curve(fpr, tpr, auc_score=None, title="ROC Curve"):
    """
    Create a plotly line chart of ROC curve
    Args:
        fpr: Array of false positive rates
        tpr: Array of true positive rates
        auc_score: AUC score (optional)
        title: Title for the plot
    Returns:
        go.Figure: Plotly figure object
    """
    # Create title with AUC if provided
    if auc_score is not None:
        title = f"{title} (AUC = {auc_score:.3f})"
    
    # Create the plot
    fig = px.line(
        x=fpr, y=tpr,
        labels=dict(x="False Positive Rate", y="True Positive Rate"),
        title=title,
        template="plotly_white"
    )
    
    # Add diagonal line
    fig.add_shape(
        type="line",
        line=dict(dash="dash", color="gray"),
        x0=0, y0=0, x1=1, y1=1
    )
    
    fig.update_layout(
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig


def plot_precision_recall_curve(precision, recall, auc_score=None, title="Precision-Recall Curve"):
    """
    Create a plotly line chart of precision-recall curve
    Args:
        precision: Array of precision values
        recall: Array of recall values
        auc_score: AUC score (optional)
        title: Title for the plot
    Returns:
        go.Figure: Plotly figure object
    """
    # Create title with AUC if provided
    if auc_score is not None:
        title = f"{title} (AUC = {auc_score:.3f})"
    
    # Create the plot
    fig = px.line(
        x=recall, y=precision,
        labels=dict(x="Recall", y="Precision"),
        title=title,
        template="plotly_white"
    )
    
    fig.update_layout(
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig


def plot_feature_distribution(df, feature, target_col="ProjectDerailmentRisk", title=None):
    """
    Create a plotly histogram of feature distribution by target value
    Args:
        df: DataFrame containing the data
        feature: Feature column name
        target_col: Target column name
        title: Title for the plot (optional)
    Returns:
        go.Figure: Plotly figure object
    """
    if title is None:
        title = f"Distribution of {feature} by Risk Level"
    
    # Check if feature is categorical or numerical
    if pd.api.types.is_numeric_dtype(df[feature]):
        # Numerical feature
        fig = px.histogram(
            df,
            x=feature,
            color=target_col,
            marginal="box",
            opacity=0.7,
            barmode="overlay",
            color_discrete_map={0: "#3498db", 1: "#e74c3c"},
            labels={target_col: "Risk Level", feature: feature},
            template="plotly_white",
            title=title,
            category_orders={target_col: [0, 1]}
        )
        
        fig.update_layout(
            xaxis_title=feature,
            yaxis_title="Count",
            legend_title_text="Risk Level",
            margin=dict(l=40, r=40, t=60, b=40)
        )
    else:
        # Categorical feature
        counts_df = df.groupby([feature, target_col]).size().reset_index(name="count")
        
        fig = px.bar(
            counts_df,
            x=feature,
            y="count",
            color=target_col,
            barmode="group",
            color_discrete_map={0: "#3498db", 1: "#e74c3c"},
            labels={target_col: "Risk Level", feature: feature, "count": "Count"},
            template="plotly_white",
            title=title,
            category_orders={target_col: [0, 1]}
        )
        
        fig.update_layout(
            xaxis_title=feature,
            yaxis_title="Count",
            legend_title_text="Risk Level",
            margin=dict(l=40, r=40, t=60, b=40)
        )
    
    return fig


def plot_scatter(df, x, y, color="ProjectDerailmentRisk", size=None, title=None):
    """
    Create a plotly scatter plot
    Args:
        df: DataFrame containing the data
        x: X-axis column name
        y: Y-axis column name
        color: Color column name
        size: Size column name (optional)
        title: Title for the plot (optional)
    Returns:
        go.Figure: Plotly figure object
    """
    if title is None:
        title = f"{y} vs {x}"
    
    # Create the plot
    fig = px.scatter(
        df,
        x=x,
        y=y,
        color=color,
        size=size,
        color_discrete_map={0: "#3498db", 1: "#e74c3c"},
        labels={color: "Risk Level"},
        template="plotly_white",
        title=title,
        category_orders={color: [0, 1]}
    )
    
    fig.update_layout(
        xaxis_title=x,
        yaxis_title=y,
        legend_title_text="Risk Level",
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig


def plot_cluster_analysis(df, target_col="ProjectDerailmentRisk", features=None, title="Project Cluster Analysis"):
    """
    Create a plotly 3D scatter plot of project clusters
    Args:
        df: DataFrame containing the data
        target_col: Target column name
        features: List of three features to use for 3D plot (optional)
        title: Title for the plot
    Returns:
        go.Figure: Plotly figure object
    """
    # Select numerical features if not provided
    if features is None:
        numerical_features = df.select_dtypes(include=["number"]).columns.tolist()
        numerical_features = [f for f in numerical_features if f != target_col]
        
        if len(numerical_features) < 3:
            raise ValueError("Need at least 3 numerical features for 3D plot")
        
        features = numerical_features[:3]
    
    # Create the plot
    fig = px.scatter_3d(
        df,
        x=features[0],
        y=features[1],
        z=features[2],
        color=target_col,
        color_discrete_map={0: "#3498db", 1: "#e74c3c"},
        labels={target_col: "Risk Level"},
        template="plotly_white",
        title=title,
        category_orders={target_col: [0, 1]}
    )
    
    fig.update_layout(
        scene=dict(
            xaxis_title=features[0],
            yaxis_title=features[1],
            zaxis_title=features[2]
        ),
        legend_title_text="Risk Level",
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig


def plot_lime_explanation(explanation, title="Feature Contributions to Risk Prediction"):
    """
    Create a plotly bar chart of LIME explanations
    Args:
        explanation: LIME explanation object or list of (feature, weight) tuples
        title: Title for the plot
    Returns:
        go.Figure: Plotly figure object
    """
    # If explanation is a LIME explanation object, extract the tuples
    if hasattr(explanation, 'as_list'):
        exp_list = explanation.as_list()
    else:
        exp_list = explanation
    
    # Create a DataFrame for plotting
    features = [item[0] for item in exp_list]
    weights = [item[1] for item in exp_list]
    
    exp_df = pd.DataFrame({
        'Feature': features,
        'Weight': weights
    })
    
    # Determine the direction of each feature's impact
    exp_df['Direction'] = exp_df['Weight'].apply(
        lambda x: "Increases Risk" if x > 0 else "Decreases Risk"
    )
    
    # Sort by absolute weight
    exp_df = exp_df.sort_values('Weight', key=abs, ascending=False)
    
    # Create the plot
    fig = px.bar(
        exp_df,
        x='Weight',
        y='Feature',
        color='Direction',
        color_discrete_map={
            "Increases Risk": "#e74c3c",
            "Decreases Risk": "#2ecc71"
        },
        title=title,
        orientation='h',
        template="plotly_white"
    )
    
    fig.update_layout(
        yaxis=dict(categoryorder='total ascending'),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig


def plot_risk_heatmap(df, x, y, title=None):
    """
    Create a plotly heatmap of risk by two categorical features
    Args:
        df: DataFrame containing the data
        x: X-axis categorical column name
        y: Y-axis categorical column name
        title: Title for the plot (optional)
    Returns:
        go.Figure: Plotly figure object
    """
    if title is None:
        title = f"Risk Heatmap by {x} and {y}"
    
    # Calculate risk rate by the two dimensions
    risk_map = df.groupby([y, x])["ProjectDerailmentRisk"].mean().reset_index()
    
    # Create a pivot table for the heatmap
    pivot_table = risk_map.pivot_table(values="ProjectDerailmentRisk", index=y, columns=x)
    
    # Create the plot
    fig = px.imshow(
        pivot_table,
        labels=dict(x=x, y=y, color="Risk Rate"),
        color_continuous_scale="RdYlGn_r",
        template="plotly_white",
        text_auto=True,
        title=title,
        aspect="auto"
    )
    
    fig.update_layout(
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    # Update colorbar
    fig.update_coloraxes(colorbar_title="Risk Rate")
    
    return fig


def plot_model_comparison(metrics_dict, metric_name="roc_auc", title=None):
    """
    Create a plotly bar chart comparing model performance
    Args:
        metrics_dict: Dictionary of model metrics {model_name: {metric_name: value}}
        metric_name: Name of the metric to compare
        title: Title for the plot (optional)
    Returns:
        go.Figure: Plotly figure object
    """
    if title is None:
        title = f"Model Comparison by {metric_name.upper()}"
    
    # Extract metric values for each model
    model_names = list(metrics_dict.keys())
    metric_values = [metrics_dict[model][metric_name] for model in model_names]
    
    # Create a DataFrame for plotting
    compare_df = pd.DataFrame({
        'Model': model_names,
        'Metric': metric_values
    })
    
    # Sort by metric value
    compare_df = compare_df.sort_values('Metric', ascending=False)
    
    # Create the plot
    fig = px.bar(
        compare_df,
        x='Model',
        y='Metric',
        color='Metric',
        color_continuous_scale=px.colors.sequential.Viridis,
        template="plotly_white",
        title=title
    )
    
    fig.update_layout(
        xaxis_title="Model",
        yaxis_title=metric_name.upper(),
        coloraxis_showscale=False,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    # Add a horizontal line at 0.5 for reference (for AUC metrics)
    if metric_name.lower() in ['auc', 'roc_auc', 'pr_auc']:
        fig.add_shape(
            type="line",
            line=dict(dash="dash", color="gray"),
            x0=-0.5, y0=0.5, x1=len(model_names) - 0.5, y1=0.5
        )
    
    return fig


def plot_risk_timeline(df, date_col="StartDate", time_unit="month", title=None):
    """
    Create a plotly line chart of risk over time
    Args:
        df: DataFrame containing the data
        date_col: Date column name
        time_unit: Time unit for aggregation ('day', 'week', 'month', 'quarter', 'year')
        title: Title for the plot (optional)
    Returns:
        go.Figure: Plotly figure object
    """
    if title is None:
        title = f"Project Risk Timeline by {time_unit.title()}"
    
    # Ensure the date column is a datetime type
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    
    # Drop rows with missing dates
    df = df.dropna(subset=[date_col])
    
    # Create a time period column based on the specified time unit
    if time_unit == "day":
        df['TimePeriod'] = df[date_col].dt.date
    elif time_unit == "week":
        df['TimePeriod'] = df[date_col].dt.to_period('W').apply(lambda x: x.start_time.date())
    elif time_unit == "month":
        df['TimePeriod'] = df[date_col].dt.to_period('M').apply(lambda x: x.start_time.date())
    elif time_unit == "quarter":
        df['TimePeriod'] = df[date_col].dt.to_period('Q').apply(lambda x: x.start_time.date())
    elif time_unit == "year":
        df['TimePeriod'] = df[date_col].dt.to_period('Y').apply(lambda x: x.start_time.date())
    else:
        raise ValueError(f"Unknown time unit: {time_unit}")
    
    # Calculate risk rate and project count by time period
    timeline_df = df.groupby('TimePeriod').agg(
        RiskRate=('ProjectDerailmentRisk', 'mean'),
        ProjectCount=('ProjectDerailmentRisk', 'count')
    ).reset_index()
    
    # Create the plot with dual Y axes
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add risk rate line
    fig.add_trace(
        go.Scatter(
            x=timeline_df['TimePeriod'],
            y=timeline_df['RiskRate'],
            name="Risk Rate",
            line=dict(color="#e74c3c", width=3),
            hovertemplate="Time Period: %{x}<br>Risk Rate: %{y:.2f}<extra></extra>"
        ),
        secondary_y=False
    )
    
    # Add project count bars
    fig.add_trace(
        go.Bar(
            x=timeline_df['TimePeriod'],
            y=timeline_df['ProjectCount'],
            name="Project Count",
            marker_color="#3498db",
            opacity=0.6,
            hovertemplate="Time Period: %{x}<br>Project Count: %{y}<extra></extra>"
        ),
        secondary_y=True
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    # Update axes
    fig.update_xaxes(title_text=f"Time Period ({time_unit.title()})")
    fig.update_yaxes(title_text="Risk Rate", secondary_y=False)
    fig.update_yaxes(title_text="Project Count", secondary_y=True)
    
    return fig


def plot_project_risks(df, project_id_col="ProjectID", project_name_col="ProjectName", top_n=20, threshold=0.5, title=None):
    """
    Create a plotly bar chart of highest risk projects
    Args:
        df: DataFrame containing the data with 'RiskProbability' column
        project_id_col: Project ID column name
        project_name_col: Project name column name
        top_n: Number of projects to display
        threshold: Risk threshold for highlighting
        title: Title for the plot (optional)
    Returns:
        go.Figure: Plotly figure object
    """
    if 'RiskProbability' not in df.columns:
        raise ValueError("DataFrame must contain 'RiskProbability' column")
    
    if title is None:
        title = f"Top {top_n} Projects by Risk Probability"
    
    # Create a DataFrame for plotting
    project_risks_df = df[[project_id_col, project_name_col, 'RiskProbability']].copy()
    
    # Sort by risk probability
    project_risks_df = project_risks_df.sort_values('RiskProbability', ascending=False).head(top_n)
    
    # Create labels that include both ID and name
    project_risks_df['Project'] = project_risks_df.apply(
        lambda row: f"{row[project_name_col]} ({row[project_id_col]})", axis=1
    )
    
    # Create a color column based on threshold
    project_risks_df['AboveThreshold'] = project_risks_df['RiskProbability'] > threshold
    
    # Create the plot
    fig = px.bar(
        project_risks_df,
        x='RiskProbability',
        y='Project',
        color='AboveThreshold',
        color_discrete_map={True: "#e74c3c", False: "#3498db"},
        labels={'RiskProbability': 'Risk Probability', 'Project': 'Project', 'AboveThreshold': 'High Risk'},
        template="plotly_white",
        title=title,
        orientation='h'
    )
    
    # Add a vertical line at the threshold
    fig.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="black",
        annotation_text=f"Threshold ({threshold})",
        annotation_position="top"
    )
    
    fig.update_layout(
        xaxis_title="Risk Probability",
        yaxis_title="",
        yaxis=dict(autorange="reversed"),  # Display highest risk at the top
        legend_title_text="Risk Level",
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig
