"""Simplified Bayesian risk analysis module for RiskLens Pro

Provides simplified Bayesian inference for risk analysis without external dependencies.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict

# Simplified Bayesian network structure for project risk
# Define risk nodes and their dependencies
RISK_NETWORK = {
    "high_risk": {  # Target node (outcome)
        "parents": ["budget_risk", "schedule_risk", "complexity_risk", "stakeholder_risk"],
        "cpt": {}  # Conditional probability table to be computed dynamically
    },
    "budget_risk": {  # Budget risk node
        "parents": ["large_budget", "unclear_requirements"],
        "cpt": {}  # Conditional probability table to be computed dynamically
    },
    "schedule_risk": {  # Schedule risk node
        "parents": ["long_duration", "team_size_risk"],
        "cpt": {}  # Conditional probability table to be computed dynamically
    },
    "complexity_risk": {  # Complexity risk node
        "parents": ["high_complexity", "novel_technology"],
        "cpt": {}  # Conditional probability table to be computed dynamically
    },
    "stakeholder_risk": {  # Stakeholder risk node
        "parents": ["multiple_stakeholders", "external_dependencies"],
        "cpt": {}  # Conditional probability table to be computed dynamically
    },
    # Root nodes (no parents)
    "large_budget": {"parents": [], "prob": 0.5},  # Prior probability
    "unclear_requirements": {"parents": [], "prob": 0.5},
    "long_duration": {"parents": [], "prob": 0.5},
    "team_size_risk": {"parents": [], "prob": 0.5},
    "high_complexity": {"parents": [], "prob": 0.5},
    "novel_technology": {"parents": [], "prob": 0.5},
    "multiple_stakeholders": {"parents": [], "prob": 0.5},
    "external_dependencies": {"parents": [], "prob": 0.5}
}

# Define conditional probability parameters for intermediate nodes
# These parameters will be used to generate CPTs
CPT_PARAMS = {
    "budget_risk": {
        "base_prob": 0.2,  # Base probability when all parents are False
        "weights": {"large_budget": 0.5, "unclear_requirements": 0.3}  # Impact weights
    },
    "schedule_risk": {
        "base_prob": 0.2,
        "weights": {"long_duration": 0.4, "team_size_risk": 0.4}
    },
    "complexity_risk": {
        "base_prob": 0.2,
        "weights": {"high_complexity": 0.6, "novel_technology": 0.3}
    },
    "stakeholder_risk": {
        "base_prob": 0.2,
        "weights": {"multiple_stakeholders": 0.4, "external_dependencies": 0.3}
    },
    "high_risk": {
        "base_prob": 0.1,
        "weights": {
            "budget_risk": 0.4, 
            "schedule_risk": 0.3, 
            "complexity_risk": 0.4, 
            "stakeholder_risk": 0.3
        }
    }
}

# Node display names for better UI
NODE_DISPLAY_NAMES = {
    "high_risk": "Project Risk",
    "budget_risk": "Budget Risk",
    "schedule_risk": "Schedule Risk",
    "complexity_risk": "Complexity Risk",
    "stakeholder_risk": "Stakeholder Risk",
    "large_budget": "Large Budget",
    "unclear_requirements": "Unclear Requirements",
    "long_duration": "Long Duration",
    "team_size_risk": "Team Size Issues",
    "high_complexity": "High Complexity",
    "novel_technology": "Novel Technology",
    "multiple_stakeholders": "Multiple Stakeholders",
    "external_dependencies": "External Dependencies"
}

# Node descriptions for tooltips and explanations
NODE_DESCRIPTIONS = {
    "high_risk": "Overall project risk considering all factors",
    "budget_risk": "Risk of budget overruns",
    "schedule_risk": "Risk of schedule delays",
    "complexity_risk": "Risk due to project complexity",
    "stakeholder_risk": "Risk related to stakeholder management",
    "large_budget": "Project has a large budget compared to organizational norms",
    "unclear_requirements": "Project requirements are not clearly defined or are still evolving",
    "long_duration": "Project has a long timeline increasing exposure to changing conditions",
    "team_size_risk": "Team size is either too small or too large for optimal performance",
    "high_complexity": "Project involves complex interconnected components or processes",
    "novel_technology": "Project uses new or unproven technologies or methods",
    "multiple_stakeholders": "Project involves many stakeholders with potentially different interests",
    "external_dependencies": "Project success depends on external factors outside direct control"
}


def generate_cpt(node, params):
    """
    Generate a conditional probability table for a node based on its parents and parameters.
    
    Args:
        node: Name of the node
        params: Dictionary with base probability and weights for parents
        
    Returns:
        dict: Conditional probability table for the node
    """
    parents = RISK_NETWORK[node]["parents"]
    if not parents:
        return {}  # Root nodes don't have CPTs
    
    # Generate all possible parent combinations
    num_parents = len(parents)
    num_combinations = 2 ** num_parents
    
    cpt = {}
    
    # For each combination of parent states
    for i in range(num_combinations):
        # Convert index to binary representation of parent states
        parent_states = format(i, f'0{num_parents}b')
        key = tuple(bool(int(parent_states[j])) for j in range(num_parents))
        
        # Calculate conditional probability based on parent states
        prob = params["base_prob"]
        
        for j, parent in enumerate(parents):
            if key[j]:  # If this parent is True
                prob += params["weights"][parent]
        
        # Ensure probability is in valid range [0, 1]
        prob = min(max(prob, 0.0), 1.0)
        
        # Store in CPT
        cpt[key] = prob
    
    return cpt


def initialize_network(df, column_mapping, risk_scores):
    """
    Initialize Bayesian network nodes with probabilities derived from data.
    
    Args:
        df: DataFrame containing project data
        column_mapping: Dictionary mapping expected column names to actual column names
        risk_scores: DataFrame with calculated risk scores
        
    Returns:
        dict: Updated network with data-driven probabilities
    """
    network = RISK_NETWORK.copy()
    
    # Update root node probabilities based on data if available
    # Map data columns to network nodes
    col_budget = column_mapping.get('budget')
    col_duration = column_mapping.get('planned_duration')
    col_complexity = column_mapping.get('complexity')
    col_team_size = column_mapping.get('team_size')
    
    if col_budget and col_budget in df.columns:
        budget_values = pd.to_numeric(df[col_budget], errors='coerce')
        budget_median = budget_values.median()
        large_budget_prob = (budget_values > budget_median).mean()
        network["large_budget"]["prob"] = float(large_budget_prob)
    
    if col_duration and col_duration in df.columns:
        duration_values = pd.to_numeric(df[col_duration], errors='coerce')
        duration_median = duration_values.median()
        long_duration_prob = (duration_values > duration_median).mean()
        network["long_duration"]["prob"] = float(long_duration_prob)
    
    if col_complexity and col_complexity in df.columns:
        # Try to convert to numeric if possible
        try:
            complexity_values = pd.to_numeric(df[col_complexity], errors='raise')
            complexity_median = complexity_values.median()
            high_complexity_prob = (complexity_values > complexity_median).mean()
        except:
            # Not numeric, try to extract from text
            complexity_text = df[col_complexity].astype(str).str.lower()
            # Check for high/very high mentions
            has_high = complexity_text.str.contains('high')
            high_complexity_prob = has_high.mean()
        
        network["high_complexity"]["prob"] = float(high_complexity_prob)
    
    if col_team_size and col_team_size in df.columns:
        team_size_values = pd.to_numeric(df[col_team_size], errors='coerce')
        team_median = team_size_values.median()
        # Team size risk is for teams too small or too large
        team_size_dev = abs(team_size_values - team_median) / team_median
        team_size_risk_prob = (team_size_dev > 0.3).mean()  # Significant deviation
        network["team_size_risk"]["prob"] = float(team_size_risk_prob)
    
    # Derive unclear requirements from complexity and project type if possible
    col_project_type = column_mapping.get('project_type')
    unclear_req_prob = 0.5  # Default
    
    if col_complexity and col_project_type:
        if col_complexity in df.columns and col_project_type in df.columns:
            # New/complex project types often have unclear requirements
            project_types = df[col_project_type].astype(str).str.lower()
            new_project = project_types.str.contains('new|transformation|novel')
            
            # Combine with complexity
            if 'high_complexity' in network and isinstance(network["high_complexity"]["prob"], float):
                unclear_req_prob = (0.6 * new_project.mean() + 0.4 * network["high_complexity"]["prob"])
    
    network["unclear_requirements"]["prob"] = float(unclear_req_prob)
    
    # Estimate multiple stakeholders and external dependencies
    # For now, use heuristics based on available data
    multi_stakeholder_prob = 0.6  # Default - most projects have multiple stakeholders
    external_dep_prob = 0.5  # Default
    
    if col_project_type in df.columns:
        project_types = df[col_project_type].astype(str).str.lower()
        # Projects with likely multiple stakeholders
        multi_types = project_types.str.contains('integration|implementation|transformation')
        multi_stakeholder_prob = max(0.4, multi_types.mean())
        
        # Projects with likely external dependencies
        ext_dep_types = project_types.str.contains('integration|interface|external|collaboration')
        external_dep_prob = max(0.3, ext_dep_types.mean())
    
    network["multiple_stakeholders"]["prob"] = float(multi_stakeholder_prob)
    network["external_dependencies"]["prob"] = float(external_dep_prob)
    
    # Generate CPTs for intermediate and outcome nodes
    for node, params in CPT_PARAMS.items():
        network[node]["cpt"] = generate_cpt(node, params)
    
    return network


def infer_probability(network, node, evidence=None):
    """
    Simplified inference to calculate probability of a node given evidence.
    
    Args:
        network: Bayesian network structure
        node: Node to calculate probability for
        evidence: Dictionary mapping nodes to their observed values (True/False)
        
    Returns:
        float: Probability of the node being True given the evidence
    """
    if evidence is None:
        evidence = {}
    
    # If node is in evidence, return its value
    if node in evidence:
        return 1.0 if evidence[node] else 0.0
    
    # If node is a root node, return its prior probability
    if not network[node]["parents"]:
        return network[node]["prob"]
    
    parents = network[node]["parents"]
    
    # First, compute the probabilities of all parents
    parent_probs = {}
    for parent in parents:
        parent_probs[parent] = infer_probability(network, parent, evidence)
    
    # Now compute the weighted average over all parent combinations
    result = 0.0
    total_weight = 0.0
    
    cpt = network[node]["cpt"]
    
    # For all possible combinations of parent values
    for i in range(2 ** len(parents)):
        # Generate parent state combination
        parent_states = format(i, f'0{len(parents)}b')
        parent_values = tuple(bool(int(parent_states[j])) for j in range(len(parents)))
        
        # Compute probability of this combination
        comb_prob = 1.0
        for j, parent in enumerate(parents):
            p = parent_probs[parent] if parent_values[j] else (1 - parent_probs[parent])
            comb_prob *= p
        
        # Get conditional probability for this combination
        cond_prob = cpt.get(parent_values, 0.5)  # Default to 0.5 if not in CPT
        
        # Add weighted contribution
        result += cond_prob * comb_prob
        total_weight += comb_prob
    
    # Normalize by total weight
    if total_weight > 0:
        result /= total_weight
    
    return result


def calculate_risk_factors(network, evidence=None):
    """
    Calculate all risk factors given evidence.
    
    Args:
        network: Bayesian network structure
        evidence: Dictionary mapping nodes to their observed values
        
    Returns:
        dict: Probabilities for all nodes in the network
    """
    if evidence is None:
        evidence = {}
        
    result = {}
    for node in network.keys():
        result[node] = infer_probability(network, node, evidence)
    
    return result


def get_most_influential_factors(network, target_node="high_risk", top_n=5):
    """
    Find the factors that most influence the target node probability.
    
    Args:
        network: Bayesian network structure
        target_node: Node to analyze influences for
        top_n: Number of top influential factors to return
        
    Returns:
        list: Tuples of (node, influence) sorted by influence
    """
    # Get baseline probability
    baseline = infer_probability(network, target_node)
    
    # Calculate influence of each root node
    influences = []
    
    # Get all root nodes
    root_nodes = [node for node in network.keys() if not network[node]["parents"]]
    
    for node in root_nodes:
        # Calculate probability when this factor is True
        evidence_true = {node: True}
        prob_true = infer_probability(network, target_node, evidence_true)
        
        # Calculate probability when this factor is False
        evidence_false = {node: False}
        prob_false = infer_probability(network, target_node, evidence_false)
        
        # Influence is the difference between these probabilities
        influence = prob_true - prob_false
        influences.append((node, influence))
    
    # Sort by absolute influence (descending)
    influences.sort(key=lambda x: abs(x[1]), reverse=True)
    
    # Return top N influences
    return influences[:top_n]


def get_network_explanation(network, evidence=None):
    """
    Generate a natural language explanation of the Bayesian network and current state.
    
    Args:
        network: Bayesian network structure
        evidence: Dictionary mapping nodes to their observed values
        
    Returns:
        str: Markdown formatted explanation
    """
    if evidence is None:
        evidence = {}
    
    # Calculate current probabilities
    probs = calculate_risk_factors(network, evidence)
    
    # Format probability as percentage
    def format_prob(p):
        return f"{p*100:.1f}%"
    
    # Generate explanation
    explanation = "### Bayesian Risk Analysis\n\n"
    
    # Overall risk assessment
    high_risk_prob = probs["high_risk"]
    risk_level = "Low" if high_risk_prob < 0.3 else "Medium" if high_risk_prob < 0.6 else "High"
    
    explanation += f"Based on the Bayesian network analysis, the overall project risk level is **{risk_level}** "   
    explanation += f"with a {format_prob(high_risk_prob)} probability of high risk.\n\n"
    
    # Explain major risk drivers (intermediate nodes)
    intermediate_nodes = [node for node in network.keys() 
                         if network[node]["parents"] and node != "high_risk"]
    
    explanation += "**Key Risk Components:**\n\n"
    
    for node in sorted(intermediate_nodes, key=lambda x: probs[x], reverse=True):
        prob = probs[node]
        component_level = "Low" if prob < 0.3 else "Medium" if prob < 0.6 else "High"
        explanation += f"- **{NODE_DISPLAY_NAMES[node]}**: {component_level} ({format_prob(prob)})\n"
    
    explanation += "\n"
    
    # Get most influential factors
    influences = get_most_influential_factors(network, "high_risk", 5)
    
    explanation += "**Most Influential Factors:**\n\n"
    
    for node, influence in influences:
        influence_pct = influence * 100
        direction = "increases" if influence > 0 else "decreases"
        explanation += f"- **{NODE_DISPLAY_NAMES[node]}**: {direction} risk by {abs(influence_pct):.1f} percentage points\n"
    
    return explanation


def create_network_visualization(network, evidence=None, highlighted_node=None):
    """
    Create a visualization of the Bayesian network.
    
    Args:
        network: Bayesian network structure
        evidence: Dictionary mapping nodes to their observed values
        highlighted_node: Node to highlight in the visualization
        
    Returns:
        Figure: Plotly figure with network visualization
    """
    if evidence is None:
        evidence = {}
    
    # Calculate current probabilities
    probs = calculate_risk_factors(network, evidence)
    
    # Define node positions for visualization
    # Layer positions
    x_positions = {
        # Outcome layer
        "high_risk": 0,
        # Intermediate layer
        "budget_risk": -1.5,
        "schedule_risk": -0.5,
        "complexity_risk": 0.5,
        "stakeholder_risk": 1.5,
        # Root layer (left side)
        "large_budget": -2.0,
        "unclear_requirements": -1.0,
        "long_duration": -0.5,
        "team_size_risk": 0.0,
        # Root layer (right side)
        "high_complexity": 0.5,
        "novel_technology": 1.0,
        "multiple_stakeholders": 1.5,
        "external_dependencies": 2.0
    }
    
    y_positions = {
        # Outcome layer
        "high_risk": 0,
        # Intermediate layer
        "budget_risk": -1.0,
        "schedule_risk": -1.0,
        "complexity_risk": -1.0,
        "stakeholder_risk": -1.0,
        # Root layer
        "large_budget": -2.0,
        "unclear_requirements": -2.0,
        "long_duration": -2.0,
        "team_size_risk": -2.0,
        "high_complexity": -2.0,
        "novel_technology": -2.0,
        "multiple_stakeholders": -2.0,
        "external_dependencies": -2.0
    }
    
    # Create lists for node visualization
    nodes = list(network.keys())
    node_x = [x_positions[node] for node in nodes]
    node_y = [y_positions[node] for node in nodes]
    node_text = [f"{NODE_DISPLAY_NAMES[node]}<br>{probs[node]*100:.1f}%" for node in nodes]
    node_info = [NODE_DESCRIPTIONS[node] for node in nodes]
    
    # Calculate node colors based on probability
    node_colors = []
    for node in nodes:
        prob = probs[node]
        if node in evidence:  # Evidence nodes should be highlighted differently
            r, g, b = (100, 100, 255) if evidence[node] else (200, 200, 255)
        else:
            # Gradient from green (low prob) to red (high prob)
            r = int(min(255, 120 + (prob * 135)))
            g = int(max(100, 255 - (prob * 155)))
            b = 100
        node_colors.append(f'rgb({r},{g},{b})')
    
    # Create node trace
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            size=30,
            color=node_colors,
            line=dict(width=2, color='rgb(50,50,50)')
        ),
        text=node_text,
        textposition="middle center",
        hovertext=node_info
    )
    
    # Create edge traces
    edge_x = []
    edge_y = []
    edge_text = []
    
    for node in network.keys():
        x0, y0 = x_positions[node], y_positions[node]
        
        for parent in network[node].get("parents", []):
            x1, y1 = x_positions[parent], y_positions[parent]
            
            # Add line coordinates
            edge_x.extend([x1, x0, None])
            edge_y.extend([y1, y0, None])
            
            # Edge text shows influence
            influence = 0
            if node in CPT_PARAMS and parent in CPT_PARAMS[node]["weights"]:
                influence = CPT_PARAMS[node]["weights"][parent]
                
            edge_text.append(f"Influence: {influence:.2f}")
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        hoverinfo='text',
        line=dict(width=1, color='#888'),
        hovertext=edge_text
    )
    
    # Create figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title="Bayesian Risk Network",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[ 
                dict(
                    text="Hover over nodes to see descriptions<br>Values show probability of each risk factor",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.01, y=0.01,
                    font=dict(size=10)
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=500
        )
    )
    
    return fig


def create_risk_impact_simulation(network, target_node="high_risk", top_n=4):
    """
    Create a chart showing the impact of different factors on the target risk.
    
    Args:
        network: Bayesian network structure
        target_node: Node to analyze impacts for
        top_n: Number of top influential factors to include
        
    Returns:
        Figure: Plotly figure with risk impact simulation
    """
    # Get baseline probability
    baseline = infer_probability(network, target_node)
    
    # Get most influential factors
    influences = get_most_influential_factors(network, target_node, top_n)
    
    # Extract nodes and influence values
    nodes = [NODE_DISPLAY_NAMES[node] for node, inf in influences]
    inf_values = [inf for node, inf in influences]
    
    # Calculate true and false probabilities
    true_probs = []
    false_probs = []
    
    for node, inf in influences:
        evidence_true = {node: True}
        evidence_false = {node: False}
        true_probs.append(infer_probability(network, target_node, evidence_true))
        false_probs.append(infer_probability(network, target_node, evidence_false))
    
    # Format as percentages
    true_pcts = [p * 100 for p in true_probs]
    false_pcts = [p * 100 for p in false_probs]
    baseline_pct = baseline * 100
    
    # Create horizontal bar chart
    fig = go.Figure()
    
    # Add baseline reference line
    fig.add_shape(
        type="line",
        x0=baseline_pct,
        y0=-0.5,
        x1=baseline_pct,
        y1=len(nodes) - 0.5,
        line=dict(color="#888", width=2, dash="dot")
    )
    
    # Add annotation for baseline
    fig.add_annotation(
        x=baseline_pct,
        y=len(nodes),
        text=f"Baseline: {baseline_pct:.1f}%",
        showarrow=False,
        font=dict(size=12),
        xanchor="center",
        yanchor="bottom"
    )
    
    # Add bars for true probabilities
    fig.add_trace(go.Bar(
        y=nodes,
        x=true_pcts,
        orientation='h',
        name='If Factor is Present',
        marker_color='#FF9800',  # Orange
        text=[f"{p:.1f}%" for p in true_pcts],
        textposition='outside',
        hovertemplate='%{y}: %{x:.1f}%<extra></extra>'
    ))
    
    # Add bars for false probabilities
    fig.add_trace(go.Bar(
        y=nodes,
        x=false_pcts,
        orientation='h',
        name='If Factor is Absent',
        marker_color='#4CAF50',  # Green
        text=[f"{p:.1f}%" for p in false_pcts],
        textposition='outside',
        hovertemplate='%{y}: %{x:.1f}%<extra></extra>'
    ))
    
    # Set layout
    fig.update_layout(
        title=f"Impact of Risk Factors on {NODE_DISPLAY_NAMES[target_node]}",
        xaxis=dict(
            title="Probability of High Risk (%)",
            range=[0, 100]
        ),
        yaxis=dict(
            title="",
            categoryorder='total ascending'
        ),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=10, r=10, t=60, b=50),
        barmode='group',
        height=400
    )
    
    return fig


def create_scenario_simulation_chart(network, scenarios):
    """
    Create a chart comparing risk probabilities across different scenarios.
    
    Args:
        network: Bayesian network structure
        scenarios: List of dictionaries with name and evidence for each scenario
        
    Returns:
        Figure: Plotly figure with scenario comparison
    """
    # Define risk nodes to compare across scenarios
    risk_nodes = ["high_risk", "budget_risk", "schedule_risk", "complexity_risk", "stakeholder_risk"]
    
    # Calculate probabilities for each scenario
    scenario_names = [s["name"] for s in scenarios]
    scenario_probs = []
    
    for scenario in scenarios:
        probs = calculate_risk_factors(network, scenario.get("evidence", {}))
        scenario_probs.append([probs[node] * 100 for node in risk_nodes])
    
    # Create grouped bar chart
    fig = go.Figure()
    
    # One trace for each risk node
    for i, node in enumerate(risk_nodes):
        fig.add_trace(go.Bar(
            x=scenario_names,
            y=[probs[i] for probs in scenario_probs],
            name=NODE_DISPLAY_NAMES[node],
            hovertemplate='%{y:.1f}%<extra></extra>'
        ))
    
    # Set layout
    fig.update_layout(
        title="Risk Profile Comparison by Scenario",
        xaxis=dict(title="Scenario"),
        yaxis=dict(title="Risk Probability (%)", range=[0, 100]),
        legend=dict(title="Risk Type"),
        barmode='group',
        height=500
    )
    
    return fig


def create_sensitivity_chart(network, target_node="high_risk"):
    """
    Create a sensitivity analysis chart showing how varying node probabilities impacts the target.
    
    Args:
        network: Bayesian network structure
        target_node: Node to analyze sensitivity for
        
    Returns:
        Figure: Plotly figure with sensitivity analysis
    """
    # Get most influential root nodes
    influences = get_most_influential_factors(network, target_node, 3)
    nodes = [node for node, _ in influences]
    
    # Range of probability values to test
    prob_range = np.linspace(0, 1, 11)
    
    # Calculate target probability for each value of each root node
    sensitivity_data = []
    
    for node in nodes:
        node_probs = []
        for p in prob_range:
            # Create evidence with just this node's probability
            tmp_network = network.copy()
            tmp_network[node]["prob"] = p
            
            # Calculate target probability
            target_prob = infer_probability(tmp_network, target_node)
            node_probs.append(target_prob * 100)  # Convert to percentage
        
        sensitivity_data.append((node, prob_range, node_probs))
    
    # Create line chart
    fig = go.Figure()
    
    # Add a trace for each node
    colors = ['#F44336', '#2196F3', '#4CAF50']  # Red, Blue, Green
    
    for i, (node, probs, values) in enumerate(sensitivity_data):
        fig.add_trace(go.Scatter(
            x=[p * 100 for p in probs],  # Convert to percentage
            y=values,
            mode='lines+markers',
            name=NODE_DISPLAY_NAMES[node],
            line=dict(color=colors[i % len(colors)], width=2),
            hovertemplate=f'{NODE_DISPLAY_NAMES[node]}<br>Value: %{{x:.0f}}%<br>Impact: %{{y:.1f}}%<extra></extra>'
        ))
    
    # Set layout
    fig.update_layout(
        title="Risk Sensitivity Analysis",
        xaxis=dict(
            title="Factor Probability (%)",
            tickvals=[p * 100 for p in prob_range],
            ticktext=[f"{p*100:.0f}%" for p in prob_range]
        ),
        yaxis=dict(
            title=f"{NODE_DISPLAY_NAMES[target_node]} Probability (%)",
            range=[0, 100]
        ),
        legend=dict(title="Risk Factor"),
        height=400
    )
    
    return fig
