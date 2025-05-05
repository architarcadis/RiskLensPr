"""Anomaly detection for RiskLens Pro

This module provides advanced anomaly detection algorithms to identify outlier projects
with unusual risk patterns that might not be caught by traditional risk scoring.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go


def detect_anomalies(df, column_mapping, method='isolation_forest', contamination=0.1):
    """Detect anomalies in project data using various algorithms
    
    Args:
        df: DataFrame containing project data
        column_mapping: Dictionary mapping expected column names to actual column names
        method: Anomaly detection method to use ('isolation_forest', 'lof', or 'ensemble')
        contamination: Expected proportion of anomalies in the dataset
        
    Returns:
        tuple: (DataFrame with anomaly scores, anomaly visualization)
    """
    # Create a copy of the dataframe to avoid modifying the original
    result_df = df.copy()
    
    # Get numeric columns from dataframe (handling missing values)
    numeric_cols = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
            # Include only if less than 20% missing values
            if df[col].isna().mean() < 0.2:
                numeric_cols.append(col)
    
    # Need at least 2 numeric columns for meaningful anomaly detection
    if len(numeric_cols) < 2:
        # Try to map important project columns if available
        mapped_cols = []
        important_columns = ['budget', 'duration', 'team_size', 'complexity']
        for col_type in important_columns:
            if col_type in column_mapping and column_mapping[col_type] in df.columns:
                # Try to convert to numeric if not already
                col_name = column_mapping[col_type]
                if not pd.api.types.is_numeric_dtype(df[col_name]):
                    try:
                        result_df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
                        if result_df[col_name].isna().mean() < 0.3:  # Allow more missing values here
                            mapped_cols.append(col_name)
                    except:
                        pass
                else:
                    mapped_cols.append(col_name)
        
        # Use mapped columns if available, otherwise return empty results
        if len(mapped_cols) >= 2:
            numeric_cols = mapped_cols
        else:
            # Not enough numeric data for anomaly detection
            result_df['anomaly_score'] = 0
            result_df['is_anomaly'] = False
            return result_df, None
    
    # Extract features for anomaly detection
    X = result_df[numeric_cols].copy()
    
    # Handle missing values by imputing with column median
    for col in X.columns:
        X[col] = X[col].fillna(X[col].median())
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply selected anomaly detection method
    if method == 'isolation_forest' or method == 'ensemble':
        # Isolation Forest for detecting anomalies
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        iso_forest.fit(X_scaled)
        if_scores = iso_forest.score_samples(X_scaled)
        # Convert to anomaly score (higher = more anomalous)
        if_anomaly_scores = -if_scores
        if_predictions = iso_forest.predict(X_scaled)
        # Convert from 1 (normal) and -1 (anomaly) to boolean
        if_is_anomaly = if_predictions == -1
    
    if method == 'lof' or method == 'ensemble':
        # Local Outlier Factor for detecting anomalies
        lof = LocalOutlierFactor(
            n_neighbors=min(20, len(X) // 2),  # Adjust neighbors based on dataset size
            contamination=contamination,
            novelty=False
        )
        lof_predictions = lof.fit_predict(X_scaled)
        # LOF doesn't have a score_samples method when novelty=False
        # Use negative_outlier_factor_ attribute instead
        lof_scores = -lof.negative_outlier_factor_
        # Convert from 1 (normal) and -1 (anomaly) to boolean
        lof_is_anomaly = lof_predictions == -1
    
    # Combine results if using ensemble method
    if method == 'ensemble':
        # Average the anomaly scores
        anomaly_scores = (if_anomaly_scores + lof_scores) / 2
        # Mark as anomaly if either method flagged it
        is_anomaly = np.logical_or(if_is_anomaly, lof_is_anomaly)
    elif method == 'isolation_forest':
        anomaly_scores = if_anomaly_scores
        is_anomaly = if_is_anomaly
    else:  # method == 'lof'
        anomaly_scores = lof_scores
        is_anomaly = lof_is_anomaly
    
    # Add results to the dataframe
    result_df['anomaly_score'] = anomaly_scores
    result_df['is_anomaly'] = is_anomaly
    
    # Create visualization of anomalies
    viz = create_anomaly_visualization(result_df, numeric_cols, column_mapping)
    
    return result_df, viz


def create_anomaly_visualization(df_with_anomalies, numeric_cols, column_mapping):
    """Create visualization of detected anomalies
    
    Args:
        df_with_anomalies: DataFrame with anomaly detection results
        numeric_cols: List of numeric columns used for anomaly detection
        column_mapping: Dictionary mapping expected column names to actual column names
        
    Returns:
        Figure: Plotly figure with anomaly visualization
    """
    # Choose the two most important features for visualization
    # First try to use budget and duration if available
    primary_cols = []
    priority_cols = ['budget', 'duration', 'complexity', 'team_size']
    
    for col_type in priority_cols:
        if col_type in column_mapping and column_mapping[col_type] in numeric_cols:
            primary_cols.append(column_mapping[col_type])
            if len(primary_cols) == 2:
                break
    
    # If we couldn't find priority columns, use the first two numeric columns
    if len(primary_cols) < 2 and len(numeric_cols) >= 2:
        primary_cols = numeric_cols[:2]
    elif len(primary_cols) < 2:
        # Not enough columns for 2D visualization
        return None
    
    # Create a scatter plot with anomaly highlighting
    x_col, y_col = primary_cols[0], primary_cols[1]
    
    # Format column names for display
    x_label = x_col.replace('_', ' ').title()
    y_label = y_col.replace('_', ' ').title()
    
    # Create scatter plot with anomalies highlighted
    fig = px.scatter(
        df_with_anomalies,
        x=x_col,
        y=y_col,
        color='is_anomaly',
        color_discrete_map={True: '#FF6900', False: '#4D4D4F'},
        hover_name=df_with_anomalies.get('ProjectName', None),
        hover_data=['anomaly_score'],
        title=f'Anomaly Detection: {x_label} vs {y_label}',
        labels={
            x_col: x_label,
            y_col: y_label,
            'is_anomaly': 'Anomaly',
            'anomaly_score': 'Anomaly Score'
        },
        template='plotly_white'
    )
    
    # Enhance the visualization
    fig.update_traces(
        marker=dict(
            size=12,
            line=dict(width=2, color='white'),
            opacity=0.7
        )
    )
    
    # Add text for top anomalies
    top_anomalies = df_with_anomalies[df_with_anomalies['is_anomaly']].sort_values('anomaly_score', ascending=False).head(5)
    
    for idx, row in top_anomalies.iterrows():
        project_name = row.get('ProjectName', f'Project {idx}')
        fig.add_annotation(
            x=row[x_col],
            y=row[y_col],
            text=project_name,
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor='#FF6900',
            ax=20,
            ay=-30,
            bgcolor='#FF6900',
            font=dict(color='white', size=10)
        )
    
    fig.update_layout(
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )
    
    return fig


def get_anomaly_insights(df_with_anomalies, column_mapping):
    """Generate insights about detected anomalies
    
    Args:
        df_with_anomalies: DataFrame with anomaly detection results
        column_mapping: Dictionary mapping expected column names to actual column names
        
    Returns:
        dict: Dictionary with anomaly insights and patterns
    """
    # Extract anomalies
    anomalies = df_with_anomalies[df_with_anomalies['is_anomaly']].copy()
    normal = df_with_anomalies[~df_with_anomalies['is_anomaly']].copy()
    
    if len(anomalies) == 0:
        return {
            'count': 0,
            'percentage': 0,
            'message': 'No anomalies detected in the dataset.'
        }
    
    # Calculate basic statistics
    anomaly_count = len(anomalies)
    anomaly_percentage = (anomaly_count / len(df_with_anomalies)) * 100
    
    # Analyze patterns in anomalies
    patterns = []
    
    # Get numeric and categorical columns
    numeric_cols = [col for col in df_with_anomalies.columns 
                   if pd.api.types.is_numeric_dtype(df_with_anomalies[col]) 
                   and not pd.api.types.is_bool_dtype(df_with_anomalies[col])
                   and col not in ['anomaly_score', 'is_anomaly']]
    
    categorical_cols = [col for col in df_with_anomalies.columns 
                       if (pd.api.types.is_object_dtype(df_with_anomalies[col]) 
                          or pd.api.types.is_categorical_dtype(df_with_anomalies[col])) 
                       and col not in ['ProjectName', 'Description']]
    
    # Check for numeric deviations
    for col in numeric_cols:
        if len(anomalies[col].dropna()) == 0 or len(normal[col].dropna()) == 0:
            continue
            
        anomaly_mean = anomalies[col].mean()
        normal_mean = normal[col].mean()
        
        # Check if there's a significant difference
        if abs(anomaly_mean - normal_mean) > normal[col].std():
            direction = 'higher' if anomaly_mean > normal_mean else 'lower'
            deviation_pct = abs((anomaly_mean / normal_mean) - 1) * 100 if normal_mean != 0 else 0
            
            if deviation_pct > 20:  # Only report significant deviations
                patterns.append({
                    'type': 'numeric',
                    'column': col,
                    'message': f"{col.replace('_', ' ').title()} is typically {direction} for anomalies ({deviation_pct:.1f}% difference)"
                })
    
    # Check for categorical patterns
    for col in categorical_cols:
        if len(anomalies[col].dropna()) == 0 or len(normal[col].dropna()) == 0:
            continue
            
        # Get value counts and normalize
        anomaly_counts = anomalies[col].value_counts(normalize=True)
        normal_counts = normal[col].value_counts(normalize=True)
        
        # Find categories that are overrepresented in anomalies
        for category, ratio in anomaly_counts.items():
            if category in normal_counts and ratio > 0.2:  # Category represents at least 20% of anomalies
                normal_ratio = normal_counts.get(category, 0)
                if ratio > (1.5 * normal_ratio):  # At least 50% overrepresented
                    patterns.append({
                        'type': 'categorical',
                        'column': col,
                        'message': f"{col.replace('_', ' ').title()} value '{category}' is overrepresented in anomalies ({ratio*100:.1f}% vs {normal_ratio*100:.1f}%)"
                    })
    
    # Get top anomalies by score
    top_anomalies = anomalies.sort_values('anomaly_score', ascending=False).head(5)
    
    # Compile insights
    insights = {
        'count': anomaly_count,
        'percentage': anomaly_percentage,
        'patterns': patterns,
        'top_anomalies': top_anomalies,
        'message': f"Detected {anomaly_count} anomalies ({anomaly_percentage:.1f}% of projects) with unusual characteristics."
    }
    
    return insights


def get_anomaly_details(project_data, anomaly_row, column_mapping):
    """Generate detailed analysis of a specific anomaly
    
    Args:
        project_data: Full DataFrame with project data
        anomaly_row: Series representing the anomalous project
        column_mapping: Dictionary mapping expected column names to actual column names
        
    Returns:
        dict: Dictionary with detailed anomaly analysis
    """
    # Extract project details
    project_id = anomaly_row.name
    project_name = anomaly_row.get('ProjectName', f'Project {project_id}')
    anomaly_score = anomaly_row.get('anomaly_score', 0)
    
    # Get normal projects for comparison
    normal_projects = project_data[~project_data['is_anomaly']].copy()
    
    # Identify the most unusual aspects of this project
    unusual_aspects = []
    
    # Check numeric columns
    numeric_cols = [col for col in project_data.columns 
                  if pd.api.types.is_numeric_dtype(project_data[col]) 
                  and not pd.api.types.is_bool_dtype(project_data[col])
                  and col not in ['anomaly_score', 'is_anomaly']]
    
    for col in numeric_cols:
        if pd.isna(anomaly_row[col]):
            continue
            
        normal_mean = normal_projects[col].mean()
        normal_std = normal_projects[col].std()
        
        if normal_std == 0:
            continue
            
        # Calculate z-score to find deviations
        z_score = (anomaly_row[col] - normal_mean) / normal_std
        
        if abs(z_score) > 2:  # More than 2 standard deviations
            direction = 'high' if z_score > 0 else 'low'
            unusual_aspects.append({
                'column': col,
                'value': anomaly_row[col],
                'normal_range': f"{normal_mean - normal_std:.2f} - {normal_mean + normal_std:.2f}",
                'z_score': z_score,
                'message': f"{col.replace('_', ' ').title()} is unusually {direction} (z-score: {z_score:.2f})"
            })
    
    # Sort by absolute z-score to highlight the most unusual aspects
    unusual_aspects.sort(key=lambda x: abs(x['z_score']), reverse=True)
    
    # Get related risk factors if available
    risk_factors = []
    risk_columns = ['risk_score', 'risk_category', 'high_risk']
    for col in risk_columns:
        if col in anomaly_row:
            risk_factors.append({
                'factor': col.replace('_', ' ').title(),
                'value': anomaly_row[col]
            })
    
    # Compile detailed analysis
    details = {
        'project_id': project_id,
        'project_name': project_name,
        'anomaly_score': anomaly_score,
        'unusual_aspects': unusual_aspects[:5],  # Top 5 most unusual aspects
        'risk_factors': risk_factors,
        'message': f"Project '{project_name}' exhibits unusual characteristics that set it apart from other projects."
    }
    
    return details
