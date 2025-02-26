import re
import numpy as np
import pandas as pd
import joblib

def extract_features_from_rtl(rtl_code, signal_name):
    """
    Extract features from the RTL code for a given signal.
    
    Parameters:
    -----------
    rtl_code : str
        Verilog RTL code
    signal_name : str
        Signal name for which features are extracted
    
    Returns:
    --------
    dict
        Dictionary containing features for the signal
    """
    # Placeholder for feature extraction logic
    features = {
        'fanin': rtl_code.count(signal_name),  # Example: count how many times signal_name appears
        'fanout': rtl_code.count(signal_name),  # Example: same for fanout (simplified logic)
        'signal_width': 32,  # Default width for simplicity
        'combinational_ops': rtl_code.count('assign'),  # Example: count how many assign statements exist
        'arithmetic_ops': rtl_code.count('+') + rtl_code.count('-'),  # Count arithmetic operations
        'mux_ops': rtl_code.count('case'),  # Count case statements for multiplexers
        'always_blocks': rtl_code.count('always'),  # Count always blocks
        'case_statements': rtl_code.count('case'),  # Count case statements
        'if_statements': rtl_code.count('if'),  # Count if statements
        'loop_constructs': rtl_code.count('for') + rtl_code.count('while'),  # Count loop constructs
        'module_complexity': len(rtl_code.splitlines()),  # Use line count as a proxy for complexity
    }
    
    return features

def determine_module_type(rtl_code):
    """
    Determine the module type based on the RTL code.
    
    Parameters:
    -----------
    rtl_code : str
        Verilog RTL code
    
    Returns:
    --------
    str
        The module type (e.g., ALU, register, etc.)
    """
    # Placeholder for module type detection logic
    if 'alu' in rtl_code.lower():
        return 'ALU'
    elif 'register' in rtl_code.lower():
        return 'Register'
    else:
        return 'Generic'

def analyze_rtl_module(rtl_code, model_path='xgb_depth_predictor.joblib'):
    """
    Analyze an entire RTL module and identify potential timing-critical signals.
    
    Parameters:
    -----------
    rtl_code : str
        Verilog RTL code
    model_path : str
        Path to the trained model
        
    Returns:
    --------
    DataFrame
        Analysis results for all signals
    """
    # Load model
    model = joblib.load(model_path)
    
    # Identify all signals in the module
    signals = []
    
    # Find input/output ports
    for match in re.finditer(r'(input|output)\s+(?:reg|wire)?\s*(?:\[(\d+):(\d+)\])?\s*(\w+)', rtl_code):
        port_type, high, low, signal_name = match.groups()
        signals.append({
            'name': signal_name,
            'type': port_type,
            'is_port': True,
            'width': int(high) - int(low) + 1 if high and low else 1
        })
    
    # Find internal signals
    for match in re.finditer(r'(reg|wire)\s+(?:\[(\d+):(\d+)\])?\s*(\w+)', rtl_code):
        sig_type, high, low, signal_name = match.groups()
        # Skip if already found as port
        if not any(s['name'] == signal_name for s in signals):
            signals.append({
                'name': signal_name,
                'type': sig_type,
                'is_port': False,
                'width': int(high) - int(low) + 1 if high and low else 1
            })
    
    # Analyze each signal
    results = []
    
    for signal in signals:
        # Skip clock and reset signals
        if signal['name'] in ['clk', 'clock', 'rst', 'reset']:
            continue
        
        # Extract features
        features = extract_features_from_rtl(rtl_code, signal['name'])
        
        # Determine module type
        module_type = determine_module_type(rtl_code)
        features['module_type'] = module_type
        
        # Add technology node and clock frequency (these would come from user input in real app)
        features['technology_node_nm'] = 28  # Default to 28nm
        features['clock_frequency_mhz'] = 500  # Default to 500MHz
        features['optimization_level'] = 2  # Default to medium optimization
        
        # Convert to DataFrame
        features_df = pd.DataFrame([features])
        
        # Make prediction
        predicted_depth = model.predict(features_df)[0]
        
        # Determine criticality level
        if predicted_depth > 10:
            criticality = 'High'
        elif predicted_depth > 7:
            criticality = 'Medium'
        else:
            criticality = 'Low'
        
        # Store results
        result = {
            'signal_name': signal['name'],
            'signal_type': signal['type'],
            'is_port': signal['is_port'],
            'signal_width': signal['width'],
            'fanin': features['fanin'],
            'fanout': features['fanout'],
            'predicted_depth': predicted_depth,
            'criticality': criticality
        }
        
        results.append(result)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Check if results_df is empty
    if results_df.empty:
        print("No signals found in the RTL code.")
        return results_df
    
    # Sort by predicted depth (descending)
    results_df = results_df.sort_values('predicted_depth', ascending=False)
    
    return results_df

def generate_timing_report(results_df, module_name, clock_period_ns=2.0):
    """
    Generate a timing report based on the analysis results.
    
    Parameters:
    -----------
    results_df : DataFrame
        Analysis results from analyze_rtl_module
    module_name : str
        Name of the module being analyzed
    clock_period_ns : float
        Clock period in nanoseconds
        
    Returns:
    --------
    str
        Formatted timing report
    """
    # Assume each logic level takes approximately 0.2ns (can be adjusted based on technology)
    logic_delay_per_level_ns = 0.2
    
    # Calculate setup slack
    results_df['delay_ns'] = results_df['predicted_depth'] * logic_delay_per_level_ns
    results_df['setup_slack_ns'] = clock_period_ns - results_df['delay_ns']
    
    # Generate report
    report = f"Timing Analysis Report for {module_name}\n"
    report += f"Clock Period: {clock_period_ns} ns\n"
    report += f"Technology Node: 28nm\n"
    report += f"Total Signals Analyzed: {len(results_df)}\n"
    report += f"Critical Signals (negative slack): {sum(results_df['setup_slack_ns'] < 0)}\n"
    report += f"Near-Critical Signals (slack < 20% of clock period): {sum((results_df['setup_slack_ns'] >= 0) & (results_df['setup_slack_ns'] < 0.2 * clock_period_ns))}\n"
    report += "\n"
    
    # Critical paths section
    report += "Critical Paths:\n"
    report += "--------------\n"
    
    critical_paths = results_df[results_df['setup_slack_ns'] < 0].sort_values('setup_slack_ns')
    
    if len(critical_paths) > 0:
        for _, path in critical_paths.iterrows():
            report += f"Signal: {path['signal_name']}\n"
            report += f"  Type: {path['signal_type']}, Width: {path['signal_width']}\n"
            report += f"  Predicted Depth: {path['predicted_depth']:.2f} levels\n"
            report += f"  Estimated Delay: {path['delay_ns']:.2f} ns\n"
            report += f"  Setup Slack: {path['setup_slack_ns']:.2f} ns (VIOLATED)\n"
            report += f"  Fan-in: {path['fanin']}, Fan-out: {path['fanout']}\n"
            report += "\n"
    else:
        report += "No critical paths found.\n\n"
    
    # Near-critical paths section
    report += "Near-Critical Paths:\n"
    report += "-------------------\n"
    
    near_critical = results_df[(results_df['setup_slack_ns'] >= 0) & 
                              (results_df['setup_slack_ns'] < 0.2 * clock_period_ns)].sort_values('setup_slack_ns')
    
    if len(near_critical) > 0:
        for _, path in near_critical.iterrows():
            report += f"Signal: {path['signal_name']}\n"
            report += f"  Type: {path['signal_type']}, Width: {path['signal_width']}\n"
            report += f"  Predicted Depth: {path['predicted_depth']:.2f} levels\n"
            report += f"  Estimated Delay: {path['delay_ns']:.2f} ns\n"
            report += f"  Setup Slack: {path['setup_slack_ns']:.2f} ns (Near-critical)\n"
            report += f"  Fan-in: {path['fanin']}, Fan-out: {path['fanout']}\n"
            report += "\n"
    else:
        report += "No near-critical paths found.\n\n"

    return report


# Example Usage:

# Assume 'rtl_code' contains your Verilog code and 'xgb_depth_predictor.joblib' is the trained ML model file path
rtl_code = """your RTL code goes here"""
results_df = analyze_rtl_module(rtl_code)

# Generate and print the timing report
if not results_df.empty:
    timing_report = generate_timing_report(results_df, "example_module")
    print(timing_report)
else:
    print("No signals found in the RTL code.")
