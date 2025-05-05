"""Heuristic-based risk scoring module for RiskLens Pro

Provides a robust rule-based system for assessing project risk without requiring
historical outcome data. Uses domain expertise and industry best practices.
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Risk categories and their score ranges
RISK_CATEGORY_LOW = "Low Risk"
RISK_CATEGORY_MEDIUM = "Medium Risk"
RISK_CATEGORY_HIGH = "High Risk"
RISK_CATEGORY_CRITICAL = "Critical Risk"

# Risk factor weights (must sum to 1.0)
RISK_WEIGHTS = {
    "budget_size": 0.15,       # Higher budget = higher risk
    "duration": 0.15,         # Longer duration = higher risk
    "complexity": 0.20,       # Higher complexity = higher risk
    "team_size": 0.10,        # Team size impact varies (too small or too large)
    "project_type": 0.15,     # Some project types inherently riskier
    "region": 0.05,           # Some regions have more regulatory complexity
    "client_type": 0.05,      # Some client types more demanding
    "multiple_stakeholders": 0.05,  # More stakeholders = higher risk
    "unclear_requirements": 0.10   # Unclear requirements = higher risk
}

# Project type risk factors (higher = riskier)
PROJECT_TYPE_RISK = {
    "new development": 0.8,
    "transformation": 0.7,
    "renovation": 0.6,
    "infrastructure": 0.6,
    "upgrade": 0.4,
    "maintenance": 0.3,
    "consulting": 0.5,
    "design": 0.5,
    "engineering": 0.6,
    "implementation": 0.7,
    "study": 0.3,
    "research": 0.4,
    "default": 0.5  # For unknown project types
}

# Region risk factors (higher = riskier)
REGION_RISK = {
    "north america": 0.4,
    "europe": 0.45,
    "asia": 0.6,
    "middle east": 0.65,
    "africa": 0.7,
    "south america": 0.55,
    "australia": 0.4,
    "default": 0.5  # For unknown regions
}

# Client type risk factors (higher = riskier)
CLIENT_TYPE_RISK = {
    "government": 0.6,   # Complex procurement, regulatory requirements
    "public": 0.55,     # Public accountability, regulatory oversight
    "private": 0.4,     # Usually more agile decision-making
    "non-profit": 0.5,  # Budget constraints but simpler governance
    "international": 0.7,  # Multiple regulatory environments, cultural differences
    "default": 0.5      # For unknown client types
}


def calculate_heuristic_risk_scores(df, column_mapping):
    """
    Calculate risk scores for each project using heuristic rules, without needing historical outcomes.
    
    Args:
        df: DataFrame containing project data
        column_mapping: Dictionary mapping expected column names to actual column names
        
    Returns:
        tuple: (DataFrame with risk scores, model metadata, risk distributions)
    """
    # Create a copy of the dataframe to avoid modifying the original
    result_df = df.copy()
    
    # Get column references from mapping, with fallbacks
    col_budget = column_mapping.get('budget')
    col_duration = column_mapping.get('planned_duration')
    col_complexity = column_mapping.get('complexity')
    col_team_size = column_mapping.get('team_size')
    col_project_type = column_mapping.get('project_type')
    col_region = column_mapping.get('region')
    col_client_type = column_mapping.get('client_type')
    
    # Initialize risk factor scores
    factor_scores = pd.DataFrame(index=df.index)
    
    # Calculate budget size risk (higher budget = higher risk)
    if col_budget is not None and col_budget in df.columns:
        budget_values = pd.to_numeric(df[col_budget], errors='coerce')
        budget_values = budget_values.fillna(budget_values.median())
        
        # Normalize budget values to a 0-1 scale using sigmoid function
        # This ensures very large budgets don't completely dominate
        budget_median = budget_values.median()
        budget_scores = 1 / (1 + np.exp(-(budget_values - budget_median) / (budget_median / 2)))
        factor_scores['budget_size'] = budget_scores
    else:
        # No budget data, use neutral value
        factor_scores['budget_size'] = 0.5
    
    # Calculate duration risk (longer duration = higher risk)
    if col_duration is not None and col_duration in df.columns:
        duration_values = pd.to_numeric(df[col_duration], errors='coerce')
        duration_values = duration_values.fillna(duration_values.median())
        
        # Normalize duration values to a 0-1 scale
        duration_median = duration_values.median()
        duration_scores = 1 / (1 + np.exp(-(duration_values - duration_median) / (duration_median / 2)))
        factor_scores['duration'] = duration_scores
    else:
        # No duration data, use neutral value
        factor_scores['duration'] = 0.5
    
    # Calculate complexity risk (if complexity field exists)
    if col_complexity is not None and col_complexity in df.columns:
        # Check if complexity is already numeric
        if pd.api.types.is_numeric_dtype(df[col_complexity]):
            # Normalize to 0-1 scale
            complexity_values = df[col_complexity]
            complexity_min = complexity_values.min()
            complexity_max = complexity_values.max()
            if complexity_max > complexity_min:
                complexity_scores = (complexity_values - complexity_min) / (complexity_max - complexity_min)
            else:
                complexity_scores = pd.Series(0.5, index=df.index)
        else:
            # Map text complexity to scores
            complexity_map = {
                'very low': 0.1,
                'low': 0.3,
                'medium': 0.5,
                'high': 0.7,
                'very high': 0.9
            }
            # Convert to lowercase for matching
            complexity_values = df[col_complexity].str.lower()
            complexity_scores = complexity_values.map(lambda x: next(
                (score for term, score in complexity_map.items() if term in str(x).lower()),
                0.5  # Default to medium if no match
            ))
        
        factor_scores['complexity'] = complexity_scores
    else:
        # Infer complexity from other factors if direct measure not available
        budget_factor = factor_scores['budget_size'] if 'budget_size' in factor_scores else pd.Series(0.5, index=df.index)
        duration_factor = factor_scores['duration'] if 'duration' in factor_scores else pd.Series(0.5, index=df.index)
        
        # Complexity often correlates with budget and duration
        factor_scores['complexity'] = (budget_factor * 0.6) + (duration_factor * 0.4)
    
    # Calculate team size risk (too small or too large teams are risky)
    if col_team_size is not None and col_team_size in df.columns:
        team_size_values = pd.to_numeric(df[col_team_size], errors='coerce')
        team_size_values = team_size_values.fillna(team_size_values.median())
        
        # Calculate ideal team size (simplified as median)
        ideal_team_size = team_size_values.median()
        
        # Teams too far from ideal (either direction) have higher risk
        # Using a V-shaped function centered on ideal size
        team_size_scores = abs(team_size_values - ideal_team_size) / (ideal_team_size * 2)
        team_size_scores = team_size_scores.clip(0, 1)  # Ensure values are in 0-1 range
        factor_scores['team_size'] = team_size_scores
    else:
        # No team size data, use neutral value
        factor_scores['team_size'] = 0.5
    
    # Calculate project type risk
    if col_project_type is not None and col_project_type in df.columns:
        project_types = df[col_project_type].str.lower() if pd.api.types.is_string_dtype(df[col_project_type]) else df[col_project_type].astype(str).str.lower()
        
        # Map project types to risk scores
        factor_scores['project_type'] = project_types.map(lambda x: next(
            (score for proj_type, score in PROJECT_TYPE_RISK.items() if proj_type in str(x).lower()),
            PROJECT_TYPE_RISK['default']  # Default if no match
        ))
    else:
        # No project type data, use neutral value
        factor_scores['project_type'] = PROJECT_TYPE_RISK['default']
    
    # Calculate region risk
    if col_region is not None and col_region in df.columns:
        regions = df[col_region].str.lower() if pd.api.types.is_string_dtype(df[col_region]) else df[col_region].astype(str).str.lower()
        
        # Map regions to risk scores
        factor_scores['region'] = regions.map(lambda x: next(
            (score for region, score in REGION_RISK.items() if region in str(x).lower()),
            REGION_RISK['default']  # Default if no match
        ))
    else:
        # No region data, use neutral value
        factor_scores['region'] = REGION_RISK['default']
    
    # Calculate client type risk
    if col_client_type is not None and col_client_type in df.columns:
        client_types = df[col_client_type].str.lower() if pd.api.types.is_string_dtype(df[col_client_type]) else df[col_client_type].astype(str).str.lower()
        
        # Map client types to risk scores
        factor_scores['client_type'] = client_types.map(lambda x: next(
            (score for client, score in CLIENT_TYPE_RISK.items() if client in str(x).lower()),
            CLIENT_TYPE_RISK['default']  # Default if no match
        ))
    else:
        # No client type data, use neutral value
        factor_scores['client_type'] = CLIENT_TYPE_RISK['default']
    
    # Calculate multiple stakeholders risk - infer from project type and client type
    factor_scores['multiple_stakeholders'] = (
        factor_scores['project_type'] * 0.5 +
        factor_scores['client_type'] * 0.5
    )
    
    # Calculate unclear requirements risk - infer from complexity and project type
    factor_scores['unclear_requirements'] = (
        factor_scores['complexity'] * 0.7 +
        factor_scores['project_type'] * 0.3
    )
    
    # Calculate weighted risk score
    risk_score = pd.Series(0.0, index=df.index)
    for factor, weight in RISK_WEIGHTS.items():
        if factor in factor_scores.columns:
            risk_score += factor_scores[factor] * weight
    
    # Normalize final score to ensure it's in 0-1 range
    risk_score = risk_score.clip(0, 1)
    
    # Add risk score and category to result dataframe
    result_df['risk_score'] = risk_score
    
    # Assign risk categories based on score thresholds
    risk_category = pd.Series(index=df.index, dtype='object')
    risk_category[risk_score < 0.4] = RISK_CATEGORY_LOW
    risk_category[(risk_score >= 0.4) & (risk_score < 0.6)] = RISK_CATEGORY_MEDIUM
    risk_category[(risk_score >= 0.6) & (risk_score < 0.8)] = RISK_CATEGORY_HIGH
    risk_category[risk_score >= 0.8] = RISK_CATEGORY_CRITICAL
    
    result_df['risk_category'] = risk_category
    
    # Generate binary risk flag (1 for high/critical, 0 for low/medium)
    result_df['high_risk'] = np.where(risk_score >= 0.6, 1, 0)
    
    # Calculate risk distributions
    risk_distribution = result_df['risk_category'].value_counts().to_dict()
    
    # Calculate factor contributions for explanation
    factor_contributions = {}
    for factor, weight in RISK_WEIGHTS.items():
        if factor in factor_scores.columns:
            factor_contributions[factor] = {
                'weight': weight,
                'values': factor_scores[factor].to_dict()
            }
    
    # Create model metadata
    model_metadata = {
        'model_type': 'heuristic',
        'risk_weights': RISK_WEIGHTS,
        'factor_contributions': factor_contributions,
        'risk_thresholds': {
            'low': 0.4,
            'medium': 0.6,
            'high': 0.8
        }
    }
    
    return result_df, model_metadata, risk_distribution


def generate_risk_factor_chart(factor_contributions, project_id=None):
    """
    Generate a chart showing the contribution of each risk factor to the overall risk score.
    
    Args:
        factor_contributions: Dictionary with factor contribution data
        project_id: Optional project ID to highlight specific project
        
    Returns:
        Figure: Plotly figure with risk factor contributions
    """
    # Extract factor weights and names
    factors = list(factor_contributions.keys())
    weights = [factor_contributions[f]['weight'] for f in factors]
    
    # If project_id is provided, get the specific values for that project
    # Otherwise use average values across all projects
    if project_id is not None and project_id in factor_contributions[factors[0]]['values']:
        values = [factor_contributions[f]['values'][project_id] for f in factors]
    else:
        # Use average values
        values = [np.mean(list(factor_contributions[f]['values'].values())) for f in factors]
    
    # Calculate weighted contributions
    weighted_values = [v * w for v, w in zip(values, weights)]
    
    # Sort by weighted contribution (descending)
    sorted_indices = np.argsort(weighted_values)[::-1]
    sorted_factors = [factors[i] for i in sorted_indices]
    sorted_values = [values[i] for i in sorted_indices]
    sorted_weights = [weights[i] for i in sorted_indices]
    sorted_weighted = [weighted_values[i] for i in sorted_indices]
    
    # Format factor names for display
    display_factors = [f.replace('_', ' ').title() for f in sorted_factors]
    
    # Create color scale based on risk level
    colors = []
    for val in sorted_values:
        if val < 0.4:
            colors.append('#4CAF50')  # Green for low risk
        elif val < 0.6:
            colors.append('#FFC107')  # Yellow for medium risk
        elif val < 0.8:
            colors.append('#FF9800')  # Orange for high risk
        else:
            colors.append('#F44336')  # Red for critical risk
    
    # Create horizontal bar chart
    fig = go.Figure()
    
    # Add bars for weighted values
    fig.add_trace(go.Bar(
        y=display_factors,
        x=sorted_weighted,
        orientation='h',
        marker_color=colors,
        text=[f"{val:.0%} × {wt:.0%} = {val*wt:.1%}" for val, wt in zip(sorted_values, sorted_weights)],
        textposition='auto',
        name='Weighted Contribution'
    ))
    
    # Set layout
    fig.update_layout(
        title="Risk Factor Contributions" + (f" for Project {project_id}" if project_id else ""),
        xaxis=dict(
            title="Contribution to Risk Score",
            tickformat=".0%"
        ),
        yaxis=dict(
            title="",
            autorange="reversed"  # Highest values at the top
        ),
        showlegend=False,
        margin=dict(l=150, r=20, t=40, b=40),
        height=400
    )
    
    return fig


def get_risk_distribution_chart(risk_distribution):
    """
    Create a pie chart showing the distribution of projects across risk categories.
    
    Args:
        risk_distribution: Dictionary with risk category counts
        
    Returns:
        Figure: Plotly pie chart
    """
    # Define risk categories in descending order of severity
    categories = [RISK_CATEGORY_CRITICAL, RISK_CATEGORY_HIGH, RISK_CATEGORY_MEDIUM, RISK_CATEGORY_LOW]
    
    # Filter to only categories that exist in the data
    categories = [cat for cat in categories if cat in risk_distribution]
    values = [risk_distribution.get(cat, 0) for cat in categories]
    
    # Define colors for risk categories
    colors = {
        RISK_CATEGORY_CRITICAL: '#F44336',  # Red
        RISK_CATEGORY_HIGH: '#FF9800',      # Orange
        RISK_CATEGORY_MEDIUM: '#FFC107',    # Yellow
        RISK_CATEGORY_LOW: '#4CAF50'        # Green
    }
    
    color_list = [colors[cat] for cat in categories]
    
    # Create pie chart
    fig = go.Figure()
    
    fig.add_trace(go.Pie(
        labels=categories,
        values=values,
        marker=dict(colors=color_list),
        textinfo='label+percent',
        insidetextorientation='radial',
        hole=0.4
    ))
    
    # Set layout
    fig.update_layout(
        title="Project Risk Distribution",
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        margin=dict(t=80, b=40, l=40, r=40),
        height=350
    )
    
    # Add total count annotation in the center
    total = sum(values)
    fig.add_annotation(
        text=f"{total}<br>Projects",
        x=0.5, y=0.5,
        font_size=16,
        showarrow=False
    )
    
    return fig


def explain_project_risk(project_data, risk_score, factor_contributions, risk_weights):
    """
    Generate a natural language explanation of project risk level and factors.
    
    Args:
        project_data: Series with project data
        risk_score: Risk score for the project
        factor_contributions: Dictionary with factor contribution data
        risk_weights: Dictionary with risk factor weights
        
    Returns:
        str: Human-readable explanation of risk determination
    """
    project_id = project_data.name
    
    # Determine risk category
    if risk_score < 0.4:
        risk_category = RISK_CATEGORY_LOW
        risk_color = "green"
    elif risk_score < 0.6:
        risk_category = RISK_CATEGORY_MEDIUM
        risk_color = "orange"
    elif risk_score < 0.8:
        risk_category = RISK_CATEGORY_HIGH
        risk_color = "red"
    else:
        risk_category = RISK_CATEGORY_CRITICAL
        risk_color = "darkred"
    
    # Get factor values for this project
    factor_values = {}
    for factor, data in factor_contributions.items():
        if project_id in data['values']:
            factor_values[factor] = data['values'][project_id]
    
    # Calculate weighted contributions
    factor_weighted = {}
    for factor, value in factor_values.items():
        if factor in risk_weights:
            factor_weighted[factor] = value * risk_weights[factor]
    
    # Sort factors by weighted contribution
    sorted_factors = sorted(factor_weighted.items(), key=lambda x: x[1], reverse=True)
    
    # Format explanation based on risk level
    explanation = f"### <span style='color:{risk_color};'>Project Risk Assessment: {risk_category} ({risk_score:.1%})</span>"
    explanation += "\n\n"
    
    # Add overview
    if risk_category in [RISK_CATEGORY_LOW, RISK_CATEGORY_MEDIUM]:
        explanation += f"This project has been assessed as having **{risk_category.lower()}** with an overall risk score of {risk_score:.1%}. "
        if sorted_factors:
            explanation += f"While the overall risk is manageable, attention should be paid to the following factors:\n\n"
    else:
        explanation += f"This project has been assessed as having **{risk_category.lower()}** with a concerning risk score of {risk_score:.1%}. "
        explanation += f"The following key risk factors require immediate attention:\n\n"
    
    # Add top contributing factors (up to 3)
    for i, (factor, weight) in enumerate(sorted_factors[:3]):
        factor_name = factor.replace('_', ' ').title()
        factor_value = factor_values[factor]
        
        # Determine factor risk level
        if factor_value < 0.4:
            factor_level = "low"
        elif factor_value < 0.6:
            factor_level = "medium"
        elif factor_value < 0.8:
            factor_level = "high"
        else:
            factor_level = "critical"
        
        # Skip factors with low contribution
        if weight < 0.05:
            continue
            
        # Format specific factor explanation
        if factor == "budget_size":
            explanation += f"**{i+1}. {factor_name}**: This project has a {factor_level} budget risk ({factor_value:.1%}), "
            explanation += f"contributing {weight:.1%} to the overall risk score.\n\n"
        
        elif factor == "duration":
            explanation += f"**{i+1}. {factor_name}**: The project duration presents {factor_level} risk ({factor_value:.1%}), "
            explanation += f"contributing {weight:.1%} to the overall risk score.\n\n"
        
        elif factor == "complexity":
            explanation += f"**{i+1}. {factor_name}**: Project complexity is rated as {factor_level} ({factor_value:.1%}), "
            explanation += f"contributing {weight:.1%} to the overall risk score.\n\n"
        
        elif factor == "team_size":
            if factor_value > 0.6:
                explanation += f"**{i+1}. {factor_name}**: The team size is non-optimal ({factor_value:.1%} risk), "
            else:
                explanation += f"**{i+1}. {factor_name}**: The team size presents {factor_level} risk ({factor_value:.1%}), "
            explanation += f"contributing {weight:.1%} to the overall risk score.\n\n"
        
        elif factor == "project_type":
            explanation += f"**{i+1}. {factor_name}**: This type of project inherently carries {factor_level} risk ({factor_value:.1%}), "
            explanation += f"contributing {weight:.1%} to the overall risk score.\n\n"
        
        elif factor == "region":
            explanation += f"**{i+1}. {factor_name}**: The regional context presents {factor_level} risk ({factor_value:.1%}), "
            explanation += f"contributing {weight:.1%} to the overall risk score.\n\n"
        
        elif factor == "client_type":
            explanation += f"**{i+1}. {factor_name}**: This client type presents {factor_level} risk ({factor_value:.1%}), "
            explanation += f"contributing {weight:.1%} to the overall risk score.\n\n"
        
        elif factor == "multiple_stakeholders":
            explanation += f"**{i+1}. Stakeholder Complexity**: The project likely involves multiple stakeholders with {factor_level} risk ({factor_value:.1%}), "
            explanation += f"contributing {weight:.1%} to the overall risk score.\n\n"
        
        elif factor == "unclear_requirements":
            explanation += f"**{i+1}. Requirements Clarity**: The requirements have {factor_level} risk of being unclear ({factor_value:.1%}), "
            explanation += f"contributing {weight:.1%} to the overall risk score.\n\n"
        
        else:
            explanation += f"**{i+1}. {factor_name}**: This factor presents {factor_level} risk ({factor_value:.1%}), "
            explanation += f"contributing {weight:.1%} to the overall risk score.\n\n"
    
    # Add conclusion based on risk level
    if risk_category == RISK_CATEGORY_LOW:
        explanation += "**Conclusion**: This project has a low overall risk profile. Standard project management practices should be sufficient, with routine monitoring."
    
    elif risk_category == RISK_CATEGORY_MEDIUM:
        explanation += "**Conclusion**: This project has a moderate risk profile that requires regular monitoring. Focus on the top risk factors identified above to prevent escalation."
    
    elif risk_category == RISK_CATEGORY_HIGH:
        explanation += "**Conclusion**: This project has a high risk profile that demands close attention. Implement specific risk mitigation strategies for each identified factor, and establish more frequent review cycles."
    
    else:  # CRITICAL
        explanation += "**Conclusion**: This project has a critical risk profile requiring immediate intervention. Consider project restructuring, additional resources, and potentially a specialized approach to mitigate the extreme risk level."
    
    return explanation


def generate_risk_mitigation_recommendations(project_data, risk_score, factor_contributions, risk_weights):
    """
    Generate specific risk mitigation recommendations based on project risk factors.
    
    Args:
        project_data: Series with project data
        risk_score: Risk score for the project
        factor_contributions: Dictionary with factor contribution data
        risk_weights: Dictionary with risk factor weights
        
    Returns:
        list: List of recommendation dictionaries with priority and text
    """
    project_id = project_data.name
    recommendations = []
    
    # Get factor values for this project
    factor_values = {}
    for factor, data in factor_contributions.items():
        if project_id in data['values']:
            factor_values[factor] = data['values'][project_id]
    
    # Calculate weighted contributions
    factor_weighted = {}
    for factor, value in factor_values.items():
        if factor in risk_weights:
            factor_weighted[factor] = value * risk_weights[factor]
    
    # Sort factors by weighted contribution
    sorted_factors = sorted(factor_weighted.items(), key=lambda x: x[1], reverse=True)
    
    # Generate targeted recommendations for top risk factors
    for factor, weight in sorted_factors:
        factor_value = factor_values[factor]
        
        # Skip factors with low contribution or low risk
        if weight < 0.05 or factor_value < 0.4:
            continue
            
        # Determine recommendation priority based on factor risk level
        priority = "medium"
        if factor_value >= 0.6:
            priority = "high"
        
        # Generate factor-specific recommendations
        if factor == "budget_size":
            if factor_value >= 0.8:
                recommendations.append({
                    "priority": "high",
                    "factor": "Budget Size",
                    "text": "Consider breaking the project into smaller, more manageable phases to reduce financial exposure."
                })
                recommendations.append({
                    "priority": "high",
                    "factor": "Budget Size",
                    "text": "Implement enhanced financial controls with mandatory review thresholds at 25%, 50%, and 75% of budget."
                })
            elif factor_value >= 0.6:
                recommendations.append({
                    "priority": "high",
                    "factor": "Budget Size",
                    "text": "Increase financial reporting frequency and implement early warning triggers for budget variances."
                })
                recommendations.append({
                    "priority": "medium",
                    "factor": "Budget Size",
                    "text": "Create a dedicated contingency reserve specifically sized for this project's risk profile."
                })
            else:
                recommendations.append({
                    "priority": priority,
                    "factor": "Budget Size",
                    "text": "Review budget allocations to ensure adequate reserves for identified risks."
                })
        
        elif factor == "duration":
            if factor_value >= 0.8:
                recommendations.append({
                    "priority": "high",
                    "factor": "Duration",
                    "text": "Break the project into shorter phases with clear milestones and go/no-go decision points."
                })
                recommendations.append({
                    "priority": "high",
                    "factor": "Duration",
                    "text": "Implement a formal schedule risk analysis using critical path and Monte Carlo simulations."
                })
            elif factor_value >= 0.6:
                recommendations.append({
                    "priority": "high",
                    "factor": "Duration",
                    "text": "Identify opportunities for overlapping activities or fast-tracking without increasing risk."
                })
                recommendations.append({
                    "priority": "medium",
                    "factor": "Duration",
                    "text": "Establish buffer management protocol with clearly defined guidelines for usage."
                })
            else:
                recommendations.append({
                    "priority": priority,
                    "factor": "Duration",
                    "text": "Implement periodic schedule reviews to identify potential delays early."
                })
        
        elif factor == "complexity":
            if factor_value >= 0.8:
                recommendations.append({
                    "priority": "high",
                    "factor": "Complexity",
                    "text": "Assign a dedicated technical architect/lead with specific expertise in this type of project."
                })
                recommendations.append({
                    "priority": "high",
                    "factor": "Complexity",
                    "text": "Develop a detailed technical risk register with specific mitigation strategies for each identified risk."
                })
            elif factor_value >= 0.6:
                recommendations.append({
                    "priority": "high",
                    "factor": "Complexity",
                    "text": "Conduct additional peer reviews at key technical decision points in the project."
                })
                recommendations.append({
                    "priority": "medium",
                    "factor": "Complexity",
                    "text": "Create proof-of-concept or prototype for the most complex technical components before full implementation."
                })
            else:
                recommendations.append({
                    "priority": priority,
                    "factor": "Complexity",
                    "text": "Schedule regular complexity reviews to ensure emerging technical challenges are addressed promptly."
                })
        
        elif factor == "team_size":
            if factor_value >= 0.8:
                if factor_weighted.get("complexity", 0) > 0.1:
                    recommendations.append({
                        "priority": "high",
                        "factor": "Team Size",
                        "text": "Evaluate team composition against technical requirements and bring in additional specialized expertise."
                    })
                else:
                    recommendations.append({
                        "priority": "high",
                        "factor": "Team Size",
                        "text": "Restructure the team into smaller, focused sub-teams with clear interfaces and responsibilities."
                    })
                recommendations.append({
                    "priority": "high",
                    "factor": "Team Size",
                    "text": "Implement enhanced team communication protocols and more structured coordination meetings."
                })
            elif factor_value >= 0.6:
                recommendations.append({
                    "priority": "high",
                    "factor": "Team Size",
                    "text": "Review team capacity against project demands and adjust staffing to optimize performance."
                })
                recommendations.append({
                    "priority": "medium",
                    "factor": "Team Size",
                    "text": "Establish clear RACI matrix to ensure proper responsibility assignment across the team."
                })
            else:
                recommendations.append({
                    "priority": priority,
                    "factor": "Team Size",
                    "text": "Monitor team workload and communication channels to identify early signs of inefficiency."
                })
        
        elif factor == "unclear_requirements":
            if factor_value >= 0.8:
                recommendations.append({
                    "priority": "high",
                    "factor": "Requirements",
                    "text": "Conduct comprehensive requirements workshops with all stakeholders to clarify and document expectations."
                })
                recommendations.append({
                    "priority": "high",
                    "factor": "Requirements",
                    "text": "Implement formal requirements traceability matrix with regular validation checkpoints."
                })
            elif factor_value >= 0.6:
                recommendations.append({
                    "priority": "high",
                    "factor": "Requirements",
                    "text": "Establish a more rigorous requirements change management process with impact analysis."
                })
                recommendations.append({
                    "priority": "medium",
                    "factor": "Requirements",
                    "text": "Create prototypes or mockups to validate understanding of key requirements with stakeholders."
                })
            else:
                recommendations.append({
                    "priority": priority,
                    "factor": "Requirements",
                    "text": "Schedule regular requirements review sessions to maintain alignment on project scope."
                })
        
        elif factor == "multiple_stakeholders":
            if factor_value >= 0.8:
                recommendations.append({
                    "priority": "high",
                    "factor": "Stakeholders",
                    "text": "Develop a formal stakeholder management plan with specific engagement strategies for each group."
                })
                recommendations.append({
                    "priority": "high",
                    "factor": "Stakeholders",
                    "text": "Establish a steering committee with representation from key stakeholder groups to streamline decision-making."
                })
            elif factor_value >= 0.6:
                recommendations.append({
                    "priority": "high",
                    "factor": "Stakeholders",
                    "text": "Create a regular stakeholder communication cadence with tailored messaging for different groups."
                })
                recommendations.append({
                    "priority": "medium",
                    "factor": "Stakeholders",
                    "text": "Implement a stakeholder influence/interest mapping to prioritize engagement efforts."
                })
            else:
                recommendations.append({
                    "priority": priority,
                    "factor": "Stakeholders",
                    "text": "Monitor stakeholder sentiment and address concerns proactively to maintain support."
                })
                
    # Add general recommendations based on overall risk level
    if risk_score >= 0.8:  # Critical risk
        recommendations.append({
            "priority": "high",
            "factor": "General",
            "text": "Consider a comprehensive project restructuring to address the critical risk profile."
        })
        recommendations.append({
            "priority": "high",
            "factor": "General",
            "text": "Implement weekly executive-level risk review meetings focused on highest impact factors."
        })
    elif risk_score >= 0.6:  # High risk
        recommendations.append({
            "priority": "high",
            "factor": "General",
            "text": "Establish a dedicated risk management team with weekly monitoring and reporting."
        })
        recommendations.append({
            "priority": "medium",
            "factor": "General",
            "text": "Develop contingency plans for the three highest risk scenarios identified."
        })
    elif risk_score >= 0.4:  # Medium risk
        recommendations.append({
            "priority": "medium",
            "factor": "General",
            "text": "Implement monthly risk reviews with focus on early warning indicators."
        })
    
    # Sort recommendations by priority
    sorted_recommendations = sorted(recommendations, key=lambda x: 0 if x["priority"] == "high" else 1)
    
    return sorted_recommendations


def format_risk_recommendations(recommendations):
    """
    Format risk mitigation recommendations as markdown with priority highlighting.
    
    Args:
        recommendations: List of recommendation dictionaries with priority and text
        
    Returns:
        str: Markdown formatted recommendations
    """
    if not recommendations:
        return "*No specific recommendations generated for this project.*"
    
    # Group recommendations by factor
    factor_groups = {}
    for rec in recommendations:
        factor = rec.get("factor", "General")
        if factor not in factor_groups:
            factor_groups[factor] = []
        factor_groups[factor].append(rec)
    
    markdown = "Based on the risk analysis, we recommend the following actions:\n\n"
    
    # Create recommendation count for ordering
    recommendation_count = 1
    
    # Create markdown for each factor group
    for factor, recs in factor_groups.items():
        # Create a styled section header with Arcadis orange
        markdown += f"<div style='margin-bottom: 15px; padding: 10px; border-left: 3px solid #FF6900;'>\n"
        markdown += f"<h5 style='margin-top: 0; color: #FF6900;'>{recommendation_count}. {factor}</h5>\n"
        
        recommendation_content = ""
        for rec in recs:
            priority = rec["priority"]
            text = rec["text"]
            recommendation_content += f"<p style='margin-bottom: 0;'>{text}</p>\n"
        
        markdown += recommendation_content
        markdown += "</div>\n\n"
        recommendation_count += 1
    
    return markdown
