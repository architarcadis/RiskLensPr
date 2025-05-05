"""Monte Carlo simulation module for RiskLens Pro

Provides advanced Monte Carlo simulation functionality for risk analysis
with no external dependencies.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def run_monte_carlo_simulation(project_data, num_simulations=5000, risk_factors=None, risk_weights=None):
    """
    Run a Monte Carlo simulation for project risk analysis
    
    Args:
        project_data: DataFrame containing project data
        num_simulations: Number of simulation runs to perform
        risk_factors: Dictionary mapping risk factors to distributions
        risk_weights: Dictionary mapping risk factors to their weights in risk calculation
        
    Returns:
        dict: Dictionary containing simulation results and visualizations
    """
    # Initialize results dictionary
    results = {
        'simulation_runs': num_simulations,
        'risk_probabilities': [],
        'factors': {},
        'scenarios': [],
    }
    
    # Default risk factors and weights if not provided
    if risk_factors is None:
        risk_factors = {
            'complexity': {'type': 'triangular', 'min': 0.1, 'mode': 0.3, 'max': 0.9},
            'schedule_pressure': {'type': 'triangular', 'min': 0.1, 'mode': 0.4, 'max': 0.8},
            'budget_pressure': {'type': 'triangular', 'min': 0.2, 'mode': 0.5, 'max': 0.9},
            'stakeholder_alignment': {'type': 'triangular', 'min': 0.1, 'mode': 0.3, 'max': 0.7},
            'team_experience': {'type': 'triangular', 'min': 0.1, 'mode': 0.4, 'max': 0.8},
            'requirement_clarity': {'type': 'triangular', 'min': 0.1, 'mode': 0.3, 'max': 0.7},
            'technical_risk': {'type': 'triangular', 'min': 0.2, 'mode': 0.5, 'max': 0.9},
        }
    
    if risk_weights is None:
        risk_weights = {
            'complexity': 0.2,
            'schedule_pressure': 0.15,
            'budget_pressure': 0.15,
            'stakeholder_alignment': 0.1,
            'team_experience': 0.15,
            'requirement_clarity': 0.1,
            'technical_risk': 0.15,
        }
    
    # Generate simulated values for each risk factor
    for factor, distribution in risk_factors.items():
        if distribution['type'] == 'triangular':
            values = np.random.triangular(
                distribution['min'],
                distribution['mode'],
                distribution['max'],
                size=num_simulations
            )
        elif distribution['type'] == 'uniform':
            values = np.random.uniform(
                distribution['min'],
                distribution['max'],
                size=num_simulations
            )
        elif distribution['type'] == 'normal':
            values = np.random.normal(
                distribution['mean'],
                distribution['std'],
                size=num_simulations
            )
            # Clip values to be within 0-1 range
            values = np.clip(values, 0, 1)
        else:
            # Default to uniform distribution
            values = np.random.uniform(0, 1, size=num_simulations)
        
        results['factors'][factor] = values
    
    # Calculate risk probabilities for each simulation
    risk_probabilities = np.zeros(num_simulations)
    
    for factor, weight in risk_weights.items():
        if factor in results['factors']:
            risk_probabilities += weight * results['factors'][factor]
    
    # Normalize to 0-1 range
    risk_probabilities = np.clip(risk_probabilities, 0, 1)
    results['risk_probabilities'] = risk_probabilities
    
    # Generate some example scenarios
    percentiles = [10, 25, 50, 75, 90]
    for p in percentiles:
        threshold = np.percentile(risk_probabilities, p)
        results['scenarios'].append({
            'name': f'{p}th Percentile',
            'threshold': threshold,
            'high_risk_rate': (risk_probabilities > threshold).mean() * 100
        })
    
    return results


def create_monte_carlo_distribution_chart(simulation_results):
    """
    Create a distribution chart of Monte Carlo simulation results
    
    Args:
        simulation_results: Results from run_monte_carlo_simulation
        
    Returns:
        Figure: Plotly figure with distribution visualization
    """
    risk_probabilities = simulation_results['risk_probabilities']
    
    # Create histogram with KDE
    fig = go.Figure()
    
    # Add histogram
    fig.add_trace(go.Histogram(
        x=risk_probabilities,
        name='Frequency',
        opacity=0.75,
        nbinsx=30,
        marker_color='#FF6900',
    ))
    
    # Add key percentile lines
    percentiles = [10, 25, 50, 75, 90]
    percentile_values = [np.percentile(risk_probabilities, p) for p in percentiles]
    percentile_colors = ['#2ecc71', '#3498db', '#9b59b6', '#f39c12', '#e74c3c']
    
    for i, (p, val, color) in enumerate(zip(percentiles, percentile_values, percentile_colors)):
        fig.add_vline(
            x=val,
            line_dash='dash',
            line_color=color,
            annotation_text=f'{p}th Percentile: {val:.2f}',
            annotation_position='top right' if i % 2 == 0 else 'top left'
        )
    
    # Update layout
    fig.update_layout(
        title="Distribution of Monte Carlo Risk Probabilities",
        xaxis_title="Risk Probability",
        yaxis_title="Frequency",
        template="plotly_white",
        showlegend=False,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    
    return fig


def create_sensitivity_analysis_chart(simulation_results, risk_weights):
    """
    Create a sensitivity analysis chart showing how each factor influences risk
    
    Args:
        simulation_results: Results from run_monte_carlo_simulation
        risk_weights: Dictionary mapping risk factors to their weights
        
    Returns:
        Figure: Plotly figure with sensitivity analysis
    """
    # Calculate correlation between each factor and overall risk
    factors = simulation_results['factors']
    risk_probabilities = simulation_results['risk_probabilities']
    
    factor_correlations = []
    for factor_name, factor_values in factors.items():
        correlation = np.corrcoef(factor_values, risk_probabilities)[0, 1]
        factor_correlations.append({
            'Factor': factor_name.replace('_', ' ').title(),
            'Correlation': correlation,
            'Weight': risk_weights.get(factor_name, 0)
        })
    
    # Create DataFrame and sort by correlation magnitude
    sensitivity_df = pd.DataFrame(factor_correlations)
    sensitivity_df['Absolute Correlation'] = sensitivity_df['Correlation'].abs()
    sensitivity_df = sensitivity_df.sort_values('Absolute Correlation', ascending=False)
    
    # Create horizontal bar chart
    fig = px.bar(
        sensitivity_df,
        x='Correlation',
        y='Factor',
        orientation='h',
        color='Correlation',
        color_continuous_scale=['#e74c3c', '#ecf0f1', '#2ecc71'],
        color_continuous_midpoint=0,
        template='plotly_white',
        title='Sensitivity Analysis: Factor Correlation with Risk',
        labels={'Correlation': 'Correlation with Risk Probability'}
    )
    
    # Add weight information
    annotations = []
    for i, row in sensitivity_df.iterrows():
        annotations.append(dict(
            x=row['Correlation'] + (0.1 if row['Correlation'] < 0 else -0.1),
            y=i,
            text=f"Weight: {row['Weight']:.2f}",
            showarrow=False,
            font=dict(size=10),
            align='left' if row['Correlation'] < 0 else 'right',
        ))
    
    fig.update_layout(
        annotations=annotations,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    
    return fig


def create_scenario_comparison_chart(simulation_results):
    """
    Create a chart comparing different risk scenarios
    
    Args:
        simulation_results: Results from run_monte_carlo_simulation
        
    Returns:
        Figure: Plotly figure with scenario comparison
    """
    scenarios = simulation_results['scenarios']
    
    # Create DataFrame for scenarios
    scenario_df = pd.DataFrame(scenarios)
    
    # Create bar chart for high risk rate
    fig = px.bar(
        scenario_df,
        x='name',
        y='high_risk_rate',
        color='high_risk_rate',
        color_continuous_scale=['#2ecc71', '#f1c40f', '#e74c3c'],
        template='plotly_white',
        title='High Risk Rate by Scenario',
        labels={
            'name': 'Scenario',
            'high_risk_rate': 'High Risk Rate (%)',
        }
    )
    
    # Add threshold values as text
    for i, row in scenario_df.iterrows():
        fig.add_annotation(
            x=i,
            y=row['high_risk_rate'] + 2,  # Adjust position above bar
            text=f"Threshold: {row['threshold']:.2f}",
            showarrow=False,
            font=dict(size=10),
        )
    
    fig.update_layout(
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis=dict(type='category'),
        coloraxis_showscale=False,
    )
    
    return fig


def create_factor_distribution_chart(simulation_results, factor_name):
    """
    Create a distribution chart for a specific risk factor
    
    Args:
        simulation_results: Results from run_monte_carlo_simulation
        factor_name: Name of the factor to visualize
        
    Returns:
        Figure: Plotly figure with factor distribution
    """
    if factor_name not in simulation_results['factors']:
        return None
    
    factor_values = simulation_results['factors'][factor_name]
    
    # Create histogram
    fig = px.histogram(
        x=factor_values,
        nbins=30,
        color_discrete_sequence=['#3498db'],
        template='plotly_white',
        title=f'Distribution of {factor_name.replace("_", " ").title()} Factor',
        labels={'x': 'Factor Value'},
    )
    
    # Add percentile lines
    percentiles = [25, 50, 75]
    for p in percentiles:
        val = np.percentile(factor_values, p)
        fig.add_vline(
            x=val,
            line_dash='dash',
            annotation_text=f'{p}th Percentile: {val:.2f}',
            annotation_position='top right',
        )
    
    fig.update_layout(
        margin=dict(l=40, r=40, t=60, b=40),
        showlegend=False,
    )
    
    return fig


def generate_monte_carlo_report(simulation_results, risk_weights):
    """
    Generate a comprehensive report from Monte Carlo simulation results
    
    Args:
        simulation_results: Results from run_monte_carlo_simulation
        risk_weights: Dictionary mapping risk factors to their weights
        
    Returns:
        str: Markdown formatted report
    """
    risk_probabilities = simulation_results['risk_probabilities']
    num_simulations = simulation_results['simulation_runs']
    
    # Calculate key statistics
    mean_risk = np.mean(risk_probabilities)
    median_risk = np.median(risk_probabilities)
    std_risk = np.std(risk_probabilities)
    p10 = np.percentile(risk_probabilities, 10)
    p90 = np.percentile(risk_probabilities, 90)
    
    # Create factor correlations for report
    factors = simulation_results['factors']
    factor_correlations = []
    
    for factor_name, factor_values in factors.items():
        correlation = np.corrcoef(factor_values, risk_probabilities)[0, 1]
        factor_correlations.append({
            'name': factor_name.replace('_', ' ').title(),
            'correlation': correlation,
            'weight': risk_weights.get(factor_name, 0)
        })
    
    # Sort factors by absolute correlation
    factor_correlations = sorted(factor_correlations, key=lambda x: abs(x['correlation']), reverse=True)
    
    # Generate the markdown report
    report = f"""
    ## Monte Carlo Simulation Report
    
    This report summarizes the results of {num_simulations:,} Monte Carlo simulations
    to analyze the uncertainty in project risk predictions.
    
    ### Key Statistics
    
    - **Mean Risk Probability:** {mean_risk:.2f}
    - **Median Risk Probability:** {median_risk:.2f}
    - **Standard Deviation:** {std_risk:.2f}
    - **10th Percentile (Best Case):** {p10:.2f}
    - **90th Percentile (Worst Case):** {p90:.2f}
    - **Risk Range (80% Confidence):** [{p10:.2f} - {p90:.2f}]
    
    ### Key Risk Factors
    
    The following risk factors have the strongest influence on overall project risk:
    
    | Risk Factor | Correlation | Weight |
    |-------------|-------------|--------|
    """
    
    # Add top 5 factors to the report
    for factor in factor_correlations[:5]:
        report += f"| {factor['name']} | {factor['correlation']:.3f} | {factor['weight']:.2f} |\n"
    
    report += f"""
    
    ### Risk Interpretation
    
    {_generate_risk_interpretation(mean_risk, std_risk, factor_correlations)}
    
    ### Recommendations
    
    {_generate_recommendations(factor_correlations, mean_risk)}
    """
    
    return report


def _generate_risk_interpretation(mean_risk, std_risk, factor_correlations):
    """
    Generate a natural language interpretation of risk based on simulation results
    
    Args:
        mean_risk: Mean risk probability
        std_risk: Standard deviation of risk probability
        factor_correlations: List of factors with correlations
        
    Returns:
        str: Markdown formatted interpretation
    """
    # Get top factors
    top_factor = factor_correlations[0]['name']
    second_factor = factor_correlations[1]['name'] if len(factor_correlations) > 1 else None
    
    # Interpret risk level
    if mean_risk < 0.3:
        risk_level = "low"
    elif mean_risk < 0.6:
        risk_level = "moderate"
    else:
        risk_level = "high"
    
    # Interpret uncertainty
    if std_risk < 0.1:
        uncertainty = "low"
        uncertainty_desc = "there is high confidence in the risk assessment"
    elif std_risk < 0.2:
        uncertainty = "moderate"
        uncertainty_desc = "there is reasonable confidence in the risk assessment"
    else:
        uncertainty = "high"
        uncertainty_desc = "there is substantial uncertainty in the risk assessment"
    
    # Create interpretation
    interpretation = f"The Monte Carlo simulation indicates an overall **{risk_level} risk level** "
    interpretation += f"with **{uncertainty} uncertainty**, meaning {uncertainty_desc}. "
    
    # Add factor-specific details
    if top_factor:
        if factor_correlations[0]['correlation'] > 0:
            interpretation += f"The **{top_factor}** factor has the strongest positive influence on risk, "
            interpretation += "meaning higher values of this factor significantly increase overall risk. "
        else:
            interpretation += f"The **{top_factor}** factor has the strongest negative influence on risk, "
            interpretation += "meaning higher values of this factor significantly decrease overall risk. "
    
    if second_factor:
        interpretation += f"The **{second_factor}** factor also plays an important role in determining risk outcomes."
    
    return interpretation


def _generate_recommendations(factor_correlations, mean_risk):
    """
    Generate recommendations based on simulation results
    
    Args:
        factor_correlations: List of factors with correlations
        mean_risk: Mean risk probability
        
    Returns:
        str: Markdown formatted recommendations
    """
    recommendations = ""
    
    # Add general recommendation based on risk level
    if mean_risk < 0.3:
        recommendations += "Given the low overall risk level, standard risk management processes should be sufficient. "
        recommendations += "Regular monitoring is recommended, with focus on early detection of changes in key risk factors.\n\n"
    elif mean_risk < 0.6:
        recommendations += "With a moderate risk level, enhanced risk management processes are recommended. "
        recommendations += "Implement regular risk reassessment and develop specific mitigation plans for key risk factors.\n\n"
    else:
        recommendations += "Given the high risk level, intensive risk management is strongly recommended. "
        recommendations += "Consider assigning dedicated risk management resources and conducting frequent reassessments.\n\n"
    
    # Add factor-specific recommendations for top 3 factors
    recommendations += "Based on the simulation results, we recommend:"    
    
    for i, factor in enumerate(factor_correlations[:3]):
        if factor['correlation'] > 0.1:  # Positive correlation (increases risk)
            recommendations += f"\n\n1. **{factor['name']}**: This factor increases risk significantly. "
            recommendations += f"Consider developing specific mitigation strategies to reduce {factor['name'].lower()}."
        elif factor['correlation'] < -0.1:  # Negative correlation (decreases risk)
            recommendations += f"\n\n1. **{factor['name']}**: This factor helps decrease risk. "
            recommendations += f"Consider strengthening this aspect further to reduce overall project risk."
    
    return recommendations
