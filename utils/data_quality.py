"""Data quality assessment module for RiskLens Pro

Provides functionality for automated data quality checking, schema detection, 
and DAMA compliance scoring.
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from fuzzywuzzy import fuzz, process

# DAMA Data Quality Dimensions
DQD_COMPLETENESS = "Completeness"
DQD_ACCURACY = "Accuracy"
DQD_CONSISTENCY = "Consistency"
DQD_TIMELINESS = "Timeliness"
DQD_VALIDITY = "Validity"
DQD_UNIQUENESS = "Uniqueness"

# Expected column groups and their priority (higher = more important)
EXPECTED_COLUMNS = {
    # Critical columns (project identification)
    "project_id": {"priority": 5, "alternatives": ["project_id", "projectid", "id", "project number", "project_number", "project_code", "project code"]},
    "project_name": {"priority": 5, "alternatives": ["project_name", "projectname", "name", "project title", "project_title", "title"]},
    
    # Project characteristics
    "project_type": {"priority": 4, "alternatives": ["project_type", "projecttype", "type", "project category", "project_category", "category"]},
    "region": {"priority": 4, "alternatives": ["region", "area", "location", "country", "geo", "geography"]},
    "client_type": {"priority": 3, "alternatives": ["client_type", "clienttype", "client category", "client_category", "customer type", "customer_type"]},
    "sector": {"priority": 3, "alternatives": ["sector", "industry", "business_unit", "business unit", "division"]},
    
    # Risk parameters
    "budget": {"priority": 5, "alternatives": ["budget", "project_budget", "planned cost", "planned_cost", "cost"]},
    "actual_cost": {"priority": 5, "alternatives": ["actual_cost", "actualcost", "final cost", "final_cost", "cost_final", "actual"]},
    "planned_duration": {"priority": 4, "alternatives": ["planned_duration", "plannedduration", "estimated duration", "estimated_duration", "duration_plan", "schedule"]},
    "actual_duration": {"priority": 4, "alternatives": ["actual_duration", "actualduration", "final duration", "final_duration", "duration_actual"]},
    "complexity": {"priority": 3, "alternatives": ["complexity", "complexity_score", "complexity score", "technical complexity", "technical_complexity"]},
    "team_size": {"priority": 3, "alternatives": ["team_size", "teamsize", "team count", "team_count", "num_resources", "resource count"]},
    
    # Risk outcome
    "overbudget": {"priority": 4, "alternatives": ["overbudget", "over_budget", "budget_overrun", "cost_overrun", "budget status"]},
    "delayed": {"priority": 4, "alternatives": ["delayed", "delay", "schedule_overrun", "late", "schedule status"]},
    "risk_score": {"priority": 3, "alternatives": ["risk_score", "riskscore", "risk rating", "risk_rating", "risk level", "risk_level"]},
}


def detect_column_schema(df):
    """
    Intelligently detect and map uploaded columns to the expected schema using fuzzy matching.
    
    Args:
        df: DataFrame containing the uploaded data
        
    Returns:
        dict: A dictionary mapping expected column names to the detected column names
    """
    column_mapping = {}
    confidence_scores = {}
    
    # Normalize column names for better matching
    normalized_columns = {col.lower().replace("_", " "): col for col in df.columns}
    
    # For each expected column, find the best match
    for expected_col, info in EXPECTED_COLUMNS.items():
        best_match = None
        best_score = 0
        
        # Try exact match first
        exact_match = None
        for alt in info["alternatives"]:
            if alt.lower() in normalized_columns:
                exact_match = normalized_columns[alt.lower()]
                break
                
        if exact_match:
            best_match = exact_match
            best_score = 100
        else:
            # Use fuzzy matching to find the closest column
            for col_norm, col_orig in normalized_columns.items():
                # Try all alternatives for this expected column
                for alt in info["alternatives"]:
                    score = fuzz.token_set_ratio(alt.lower(), col_norm)
                    if score > best_score:
                        best_score = score
                        best_match = col_orig
        
        # Only consider matches with score above threshold, weighted by priority
        score_threshold = 90 - (info["priority"] * 5)  # Lower threshold for high priority columns
        if best_score >= score_threshold:
            column_mapping[expected_col] = best_match
            confidence_scores[expected_col] = best_score
        else:
            column_mapping[expected_col] = None
            confidence_scores[expected_col] = 0
            
    return column_mapping, confidence_scores


def assess_data_quality(df, column_mapping):
    """
    Assess the quality of the data across DAMA dimensions.
    
    Args:
        df: DataFrame to analyze
        column_mapping: Dictionary mapping expected columns to actual columns
        
    Returns:
        dict: Dictionary of quality metrics for each DAMA dimension
    """
    quality_scores = {}
    column_scores = {}
    data_issues = []
    total_records = len(df)
    
    # Only process columns that were mapped
    mapped_columns = {k: v for k, v in column_mapping.items() if v is not None}
    
    # Filter out columns that don't exist in the dataframe
    valid_columns = {k: v for k, v in mapped_columns.items() if v in df.columns}
    
    # Calculate completeness score
    completeness_scores = {}
    for expected_col, actual_col in valid_columns.items():
        if actual_col is not None:
            missing_count = df[actual_col].isna().sum()
            completeness = 1 - (missing_count / total_records)
            completeness_scores[expected_col] = completeness
            
            if missing_count > 0:
                data_issues.append({
                    "dimension": DQD_COMPLETENESS,
                    "column": actual_col,
                    "issue": f"Missing {missing_count} values ({missing_count/total_records:.1%})",
                    "impact": "high" if EXPECTED_COLUMNS[expected_col]["priority"] >= 4 else "medium"
                })
    
    # Calculate validity score (data types, ranges)
    validity_scores = {}
    for expected_col, actual_col in valid_columns.items():
        if actual_col is not None:
            # Check appropriate data types
            is_valid = True
            if expected_col in ['budget', 'actual_cost']:
                # Should be numeric
                try:
                    if not pd.api.types.is_numeric_dtype(df[actual_col]):
                        is_valid = False
                        data_issues.append({
                            "dimension": DQD_VALIDITY,
                            "column": actual_col,
                            "issue": "Column should be numeric",
                            "impact": "high"
                        })
                except:
                    is_valid = False
            
            # Check ranges for numeric columns
            if is_valid and pd.api.types.is_numeric_dtype(df[actual_col]):
                # Check for extreme outliers using IQR
                q1 = df[actual_col].quantile(0.25)
                q3 = df[actual_col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - (3 * iqr)
                upper_bound = q3 + (3 * iqr)
                outliers = df[(df[actual_col] < lower_bound) | (df[actual_col] > upper_bound)]
                outlier_pct = len(outliers) / total_records
                
                if outlier_pct > 0.05:  # More than 5% outliers
                    is_valid = False
                    data_issues.append({
                        "dimension": DQD_VALIDITY,
                        "column": actual_col,
                        "issue": f"Contains {len(outliers)} outliers ({outlier_pct:.1%})",
                        "impact": "medium"
                    })
            
            validity_scores[expected_col] = 1.0 if is_valid else 0.5
    
    # Calculate consistency score
    consistency_scores = {}
    for expected_col, actual_col in valid_columns.items():
        if actual_col is not None:
            consistency = 1.0  # Default as consistent
            
            # Check for potential inconsistencies in categorical columns
            if pd.api.types.is_string_dtype(df[actual_col]):
                # Check for similar values that might be typos
                value_counts = df[actual_col].value_counts()
                frequent_values = value_counts[value_counts > 1].index.tolist()
                
                # Skip if too many unique values (likely not categorical)
                if len(frequent_values) < 20:
                    similar_groups = []
                    for i, val1 in enumerate(frequent_values):
                        for val2 in frequent_values[i+1:]:
                            similarity = fuzz.ratio(str(val1).lower(), str(val2).lower())
                            if similarity > 80:  # Very similar strings
                                similar_groups.append((val1, val2, similarity))
                    
                    if similar_groups:
                        similarity_impact = min(len(similar_groups) / 10, 1.0)  # Scale based on number of similar groups
                        consistency = 1.0 - (similarity_impact * 0.5)  # Reduce score by up to 50%
                        
                        data_issues.append({
                            "dimension": DQD_CONSISTENCY,
                            "column": actual_col,
                            "issue": f"Contains {len(similar_groups)} potentially inconsistent value groups",
                            "impact": "medium"
                        })
            
            consistency_scores[expected_col] = consistency
    
    # Calculate uniqueness score
    uniqueness_scores = {}
    for expected_col, actual_col in valid_columns.items():
        if actual_col is not None:
            # Uniqueness only applies to certain columns
            if expected_col in ['project_id']:
                duplicates = df[actual_col].duplicated().sum()
                uniqueness = 1.0 - (duplicates / total_records)
                uniqueness_scores[expected_col] = uniqueness
                
                if duplicates > 0:
                    data_issues.append({
                        "dimension": DQD_UNIQUENESS,
                        "column": actual_col,
                        "issue": f"Contains {duplicates} duplicate values",
                        "impact": "high"
                    })
            else:
                uniqueness_scores[expected_col] = 1.0  # Not applicable for this column
    
    # Calculate timeliness score (simplified - based on completeness)
    timeliness_scores = {k: completeness_scores.get(k, 0) for k in valid_columns.keys()}
    
    # Combine column-level scores into dimension scores
    def weighted_average(scores_dict):
        total_weight = 0
        weighted_sum = 0
        for col, score in scores_dict.items():
            if col in EXPECTED_COLUMNS:
                weight = EXPECTED_COLUMNS[col]["priority"]
                weighted_sum += score * weight
                total_weight += weight
        return weighted_sum / total_weight if total_weight > 0 else 0
    
    # Calculate overall dimension scores
    quality_scores[DQD_COMPLETENESS] = weighted_average(completeness_scores)
    quality_scores[DQD_VALIDITY] = weighted_average(validity_scores)
    quality_scores[DQD_CONSISTENCY] = weighted_average(consistency_scores)
    quality_scores[DQD_UNIQUENESS] = weighted_average(uniqueness_scores)
    quality_scores[DQD_TIMELINESS] = weighted_average(timeliness_scores)
    
    # Calculate column scores as average across dimensions
    for col in valid_columns.keys():
        dimensions = []
        scores = []
        
        if col in completeness_scores:
            dimensions.append(DQD_COMPLETENESS)
            scores.append(completeness_scores[col])
        
        if col in validity_scores:
            dimensions.append(DQD_VALIDITY)
            scores.append(validity_scores[col])
            
        if col in consistency_scores:
            dimensions.append(DQD_CONSISTENCY)
            scores.append(consistency_scores[col])
            
        if col in uniqueness_scores:
            dimensions.append(DQD_UNIQUENESS)
            scores.append(uniqueness_scores[col])
            
        if col in timeliness_scores:
            dimensions.append(DQD_TIMELINESS)
            scores.append(timeliness_scores[col])
        
        if scores:  # Only calculate if we have scores
            column_scores[col] = {
                "dimensions": dimensions,
                "scores": scores,
                "overall": sum(scores) / len(scores),
                "priority": EXPECTED_COLUMNS[col]["priority"]
            }
    
    # Sort issues by impact
    sorted_issues = sorted(data_issues, key=lambda x: 0 if x["impact"] == "high" else 1 if x["impact"] == "medium" else 2)
    
    # Calculate overall data quality score
    overall_quality = sum(quality_scores.values()) / len(quality_scores)
    
    return {
        "dimension_scores": quality_scores,
        "column_scores": column_scores,
        "overall_quality": overall_quality,
        "data_issues": sorted_issues
    }


def get_data_quality_chart(quality_scores):
    """
    Create a radar chart for DAMA data quality dimensions.
    
    Args:
        quality_scores: Dictionary with quality scores for each dimension
        
    Returns:
        Figure: Plotly radar chart figure
    """
    dama_dimensions = list(quality_scores["dimension_scores"].keys())
    scores = [quality_scores["dimension_scores"][dim] for dim in dama_dimensions]
    
    # Create radar chart
    fig = go.Figure()
    
    # Add the radar chart trace
    fig.add_trace(go.Scatterpolar(
        r=scores,
        theta=dama_dimensions,
        fill='toself',
        fillcolor='rgba(255, 105, 0, 0.2)',  # Arcadis orange with transparency
        line=dict(color='#FF6900'),  # Arcadis orange for line
        name='Data Quality'
    ))
    
    # Add a reference circle at 80%
    theta_reference = dama_dimensions + [dama_dimensions[0]]
    r_reference = [0.8] * (len(dama_dimensions) + 1)
    fig.add_trace(go.Scatterpolar(
        r=r_reference,
        theta=theta_reference,
        mode='lines',
        line=dict(color='#4D4D4F', dash='dash', width=1),  # Arcadis gray dashed line
        name='Target (80%)'
    ))
    
    # Set layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                ticktext=['0%', '20%', '40%', '60%', '80%', '100%'],
                griddash='dash',
                gridwidth=0.5,
                gridcolor='lightgray',
                linewidth=0.5,
                linecolor='lightgray',
                bgcolor='white'
            ),
            angularaxis=dict(
                gridcolor='lightgray',
                griddash='dash',
                gridwidth=0.5,
                linewidth=0.5,
                linecolor='lightgray'
            )
        ),
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.1, xanchor='center', x=0.5),
        margin=dict(t=50, b=50, l=50, r=50),
        height=350
    )
    
    return fig


def get_column_quality_chart(quality_scores):
    """
    Create a horizontal bar chart showing quality scores by column.
    
    Args:
        quality_scores: Dictionary with column-level quality scores
        
    Returns:
        Figure: Plotly bar chart figure
    """
    # Sort columns by priority and quality score
    columns = []
    scores = []
    priorities = []
    
    for col, data in quality_scores["column_scores"].items():
        columns.append(col)
        scores.append(data["overall"])
        priorities.append(data["priority"])
    
    # Create DataFrame for sorting
    df = pd.DataFrame({
        "column": columns,
        "score": scores,
        "priority": priorities
    })
    
    # Sort by priority (desc) then by score (asc)
    df = df.sort_values(["priority", "score"], ascending=[False, True])
    
    # Define colors based on score
    colors = []
    for score in df["score"]:
        if score >= 0.8:
            colors.append("#4CAF50")  # Green for good
        elif score >= 0.5:
            colors.append("#FFC107")  # Yellow for warning
        else:
            colors.append("#F44336")  # Red for poor
    
    # Create bar chart
    fig = go.Figure()
    
    # Add the main bars
    fig.add_trace(go.Bar(
        x=df["score"],
        y=df["column"],
        orientation='h',
        marker_color=colors,
        text=[f"{score:.0%}" for score in df["score"]],
        textposition='auto',
        hoverinfo='text',
        hovertext=[f"{col}: {score:.1%}" for col, score in zip(df["column"], df["score"])]
    ))
    
    # Add a vertical line at 80%
    fig.add_shape(type="line",
        x0=0.8, y0=-0.5, x1=0.8, y1=len(df)-0.5,
        line=dict(color="#4D4D4F", width=1, dash="dash")
    )
    
    # Add "Target" annotation
    fig.add_annotation(
        x=0.8, y=len(df),
        text="Target (80%)",
        showarrow=False,
        font=dict(size=10),
        xanchor="center",
        yanchor="bottom"
    )
    
    # Set layout
    fig.update_layout(
        title="Data Quality by Field",
        xaxis=dict(
            title="Quality Score",
            range=[0, 1],
            tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            ticktext=['0%', '20%', '40%', '60%', '80%', '100%']
        ),
        yaxis=dict(
            title="",
            autorange="reversed"  # Highest values at the top
        ),
        showlegend=False,
        margin=dict(l=120, r=20, t=40, b=40),
        height=max(350, 30 * len(df))
    )
    
    return fig


def get_data_quality_summary(quality_scores):
    """
    Generate a text summary of data quality assessment.
    
    Args:
        quality_scores: Dictionary with quality scores
        
    Returns:
        str: Markdown formatted text summary
    """
    overall = quality_scores["overall_quality"]
    dimension_scores = quality_scores["dimension_scores"]
    issues = quality_scores["data_issues"]
    
    # Determine overall rating
    rating = "Excellent" if overall >= 0.9 else "Good" if overall >= 0.8 else "Fair" if overall >= 0.6 else "Poor"
    
    # Count high and medium impact issues
    high_impact = sum(1 for issue in issues if issue["impact"] == "high")
    medium_impact = sum(1 for issue in issues if issue["impact"] == "medium")
    
    # Create summary
    summary = f"### Data Quality Assessment\n\n"
    summary += f"**Overall Quality:** {rating} ({overall:.1%})\n\n"
    
    # Add dimension ratings
    summary += "**Dimension Scores:**\n"
    for dim, score in dimension_scores.items():
        dim_rating = "✅ Excellent" if score >= 0.9 else "✓ Good" if score >= 0.8 else "⚠️ Fair" if score >= 0.6 else "❌ Poor"
        summary += f"- {dim}: {score:.1%} ({dim_rating})\n"
    
    # Add issue summary
    if issues:
        summary += f"\n**Issues Detected:** {len(issues)} total ({high_impact} high impact, {medium_impact} medium impact)\n"
        
        # List high impact issues
        if high_impact > 0:
            summary += "\n**High Impact Issues:**\n"
            for issue in [i for i in issues if i["impact"] == "high"]:
                summary += f"- {issue['column']}: {issue['issue']} ({issue['dimension']})\n"
    else:
        summary += "\n**Issues Detected:** None - excellent data quality!\n"
    
    return summary


def get_data_quality_recommendations(quality_scores, column_mapping):
    """
    Generate recommendations for improving data quality.
    
    Args:
        quality_scores: Dictionary with quality scores
        column_mapping: Dictionary mapping expected columns to actual columns
        
    Returns:
        str: Markdown formatted recommendations
    """
    recommendations = []
    issues = quality_scores["data_issues"]
    missing_columns = [col for col, mapped in column_mapping.items() if mapped is None]
    
    # Check for missing critical columns
    critical_missing = [col for col in missing_columns 
                      if col in EXPECTED_COLUMNS and EXPECTED_COLUMNS[col]["priority"] >= 4]
    if critical_missing:
        recommendations.append({
            "priority": "high",
            "text": f"Add the following critical fields to your dataset: {', '.join(critical_missing)}"
        })
    
    # Check for missing non-critical columns
    non_critical_missing = [col for col in missing_columns 
                          if col in EXPECTED_COLUMNS and EXPECTED_COLUMNS[col]["priority"] < 4]
    if non_critical_missing:
        recommendations.append({
            "priority": "medium",
            "text": f"Consider adding these fields for better analysis: {', '.join(non_critical_missing)}"
        })
    
    # Address high impact issues
    for issue in [i for i in issues if i["impact"] == "high"]:
        if issue["dimension"] == DQD_COMPLETENESS:
            recommendations.append({
                "priority": "high",
                "text": f"Fill in missing values for {issue['column']}"
            })
        elif issue["dimension"] == DQD_UNIQUENESS:
            recommendations.append({
                "priority": "high",
                "text": f"Remove duplicate values in {issue['column']}"
            })
        elif issue["dimension"] == DQD_VALIDITY:
            recommendations.append({
                "priority": "high",
                "text": f"Fix data type or invalid values in {issue['column']}"
            })
    
    # Address medium impact issues
    consistency_issues = [i for i in issues if i["impact"] == "medium" and i["dimension"] == DQD_CONSISTENCY]
    if consistency_issues:
        columns = list(set(i["column"] for i in consistency_issues))
        recommendations.append({
            "priority": "medium",
            "text": f"Standardize inconsistent values in: {', '.join(columns)}"
        })
    
    # Sort recommendations by priority
    sorted_recommendations = sorted(recommendations, key=lambda x: 0 if x["priority"] == "high" else 1)
    
    # Format recommendations as markdown
    recommendation_text = "### Recommendations to Improve Data Quality\n\n"
    
    if sorted_recommendations:
        for i, rec in enumerate(sorted_recommendations):
            priority_icon = "🔴" if rec["priority"] == "high" else "🟠"
            recommendation_text += f"{priority_icon} {rec['text']}\n\n"
    else:
        recommendation_text += "✅ No specific recommendations - data quality is excellent!\n"
    
    return recommendation_text
