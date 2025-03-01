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
        # Add the missing features that the model expects
        'is_registered': 1 if re.search(r'reg\s+.*' + signal_name, rtl_code) is not None else 0,  # Check if signal is a register
        'is_output': 1 if re.search(r'output\s+.*' + signal_name, rtl_code) is not None else 0,  # Check if signal is an output
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
    elif 'multiplier' in rtl_code.lower() or 'mult' in rtl_code.lower():
        return 'Multiplier'
    elif 'divider' in rtl_code.lower() or 'div' in rtl_code.lower():
        return 'Divider'
    elif 'adder' in rtl_code.lower():
        return 'Adder'
    elif 'mux' in rtl_code.lower() or 'selector' in rtl_code.lower():
        return 'Multiplexer'
    elif 'fifo' in rtl_code.lower():
        return 'FIFO'
    elif 'memory' in rtl_code.lower() or 'ram' in rtl_code.lower() or 'rom' in rtl_code.lower():
        return 'Memory'
    elif 'fsm' in rtl_code.lower() or 'state_machine' in rtl_code.lower():
        return 'FSM'
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
    
    # Extract module name
    module_name_match = re.search(r'module\s+(\w+)', rtl_code)
    module_name = module_name_match.group(1) if module_name_match else "unknown_module"
    
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
            'criticality': criticality,
            'module_type': module_type,
            'combinational_ops': features['combinational_ops'],
            'arithmetic_ops': features['arithmetic_ops'],
            'mux_ops': features['mux_ops']
        }
        
        results.append(result)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Check if results_df is empty
    if results_df.empty:
        print("No signals found in the RTL code.")
        return results_df, module_name
    
    # Sort by predicted depth (descending)
    results_df = results_df.sort_values('predicted_depth', ascending=False)
    
    return results_df, module_name

def generate_timing_recommendations(signal_info, predicted_depth, slack_ns, module_type):
    """
    Generate recommendations for fixing timing violations.
    
    Parameters:
    -----------
    signal_info : dict
        Information about the signal
    predicted_depth : float
        Predicted combinational depth
    slack_ns : float
        Setup slack in nanoseconds
    module_type : str
        Type of module containing the signal
        
    Returns:
    --------
    list
        List of recommendations
    """
    recommendations = []
    
    # General recommendations based on slack violation severity
    if slack_ns < -1.0:
        recommendations.append("CRITICAL: This path has severe timing violations and needs immediate architectural changes.")
    
    # Recommendations based on combinational depth
    if predicted_depth > 12:
        recommendations.append(f"Break down combinational logic path (depth: {predicted_depth:.1f}) by inserting pipeline registers.")
        recommendations.append("Consider adding at least {0} pipeline stages to meet timing.".format(int(predicted_depth / 6) + 1))
    elif predicted_depth > 8:
        recommendations.append(f"Add pipeline registers to reduce combinational depth (currently {predicted_depth:.1f}).")
    
    # Module-specific recommendations
    if module_type == "ALU":
        if predicted_depth > 7:
            recommendations.append("Consider using a carry-lookahead or carry-select architecture for arithmetic operations.")
        if signal_info.get('arithmetic_ops', 0) > 2:
            recommendations.append("Break complex arithmetic operations into multiple cycles.")
    
    elif module_type == "Multiplier":
        recommendations.append("Consider using DSP blocks or specialized multiplier IP cores.")
        recommendations.append("Implement Booth's algorithm or Wallace tree multiplier for better performance.")
    
    elif module_type == "Divider":
        recommendations.append("Replace division with shift operations where possible.")
        recommendations.append("Implement multi-cycle division logic with proper handshaking.")
    
    # Recommendations based on fanout
    if signal_info.get('fanout', 0) > 10:
        recommendations.append(f"High fanout detected ({signal_info.get('fanout')}). Register and duplicate the signal to reduce load.")
    
    # Recommendations based on signal width
    if signal_info.get('signal_width', 0) > 32:
        recommendations.append(f"Wide signal ({signal_info.get('signal_width')} bits) detected. Consider processing in chunks.")
    
    # Add generic recommendations if no specific ones apply
    if not recommendations:
        recommendations.append("Consider retiming or resynthesizing the logic for better timing.")
        recommendations.append("Review critical path for optimization opportunities.")
    
    return recommendations

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
    
    # Add overall module summary and recommendations
    report += "Module Level Analysis:\n"
    report += "---------------------\n"
    avg_depth = results_df['predicted_depth'].mean()
    max_depth = results_df['predicted_depth'].max()
    report += f"Average Combinational Depth: {avg_depth:.2f} levels\n"
    report += f"Maximum Combinational Depth: {max_depth:.2f} levels\n"
    
    if max_depth > 10:
        report += "Overall Assessment: This module has deep combinational paths that likely cause timing violations.\n"
        report += "Module Recommendations:\n"
        if 'ALU' in module_name.upper() or results_df['module_type'].str.contains('ALU').any():
            report += "  - Consider pipelining the ALU operations\n"
            report += "  - Separate critical arithmetic operations\n"
        else:
            report += "  - Add pipeline registers to break long combinational paths\n"
            report += "  - Review complex operations for multi-cycle implementation\n"
    elif max_depth > 7:
        report += "Overall Assessment: This module has moderate combinational depth that may cause timing challenges.\n"
        report += "Module Recommendations:\n"
        report += "  - Optimize critical paths by simplifying logic\n"
        report += "  - Consider retiming strategies for balanced paths\n"
    else:
        report += "Overall Assessment: This module has reasonable combinational depth with low risk of timing violations.\n"
    
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
            
            # Generate recommendations
            signal_info = {
                'signal_width': path['signal_width'],
                'fanout': path['fanout'],
                'arithmetic_ops': path.get('arithmetic_ops', 0),
                'combinational_ops': path.get('combinational_ops', 0),
                'mux_ops': path.get('mux_ops', 0)
            }
            
            recommendations = generate_timing_recommendations(
                signal_info, 
                path['predicted_depth'], 
                path['setup_slack_ns'],
                path['module_type']
            )
            
            report += "  Recommendations:\n"
            for i, rec in enumerate(recommendations, 1):
                report += f"    {i}. {rec}\n"
            
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
            
            # Generate recommendations
            signal_info = {
                'signal_width': path['signal_width'],
                'fanout': path['fanout'],
                'arithmetic_ops': path.get('arithmetic_ops', 0),
                'combinational_ops': path.get('combinational_ops', 0),
                'mux_ops': path.get('mux_ops', 0)
            }
            
            recommendations = generate_timing_recommendations(
                signal_info, 
                path['predicted_depth'], 
                path['setup_slack_ns'],
                path['module_type']
            )
            
            report += "  Recommendations:\n"
            for i, rec in enumerate(recommendations, 1):
                report += f"    {i}. {rec}\n"
                
            report += "\n"
    else:
        report += "No near-critical paths found.\n\n"
    
    # Add section for visualizing combinational depth distribution
    report += "Combinational Depth Distribution:\n"
    report += "--------------------------------\n"
    depth_bins = [0, 3, 5, 7, 10, 15, float('inf')]
    depth_labels = ['0-3', '3-5', '5-7', '7-10', '10-15', '15+']
    
    # Count signals in each bin
    bin_counts = [0] * len(depth_labels)
    for depth in results_df['predicted_depth']:
        for i, (lower, upper) in enumerate(zip(depth_bins[:-1], depth_bins[1:])):
            if lower <= depth < upper:
                bin_counts[i] += 1
                break
    
    # Create a simple ASCII histogram
    max_count = max(bin_counts)
    scale_factor = 50 / max_count if max_count > 0 else 1
    
    for i, (label, count) in enumerate(zip(depth_labels, bin_counts)):
        bar = '█' * int(count * scale_factor)
        report += f"{label:<5}: {bar} ({count} signals)\n"
    
    report += "\n"
    
    # Add summary of recommendations section
    report += "Summary of Key Recommendations:\n"
    report += "-----------------------------\n"
    
    if max_depth > 10:
        report += "1. Major architectural changes needed to meet timing requirements\n"
        report += "2. Add pipeline stages to break critical paths\n"
        report += "3. Consider using specialized IP cores for complex operations\n"
    elif max_depth > 7:
        report += "1. Optimize critical paths through retiming and logic restructuring\n"
        report += "2. Address high-fanout signals\n"
        report += "3. Review complex operations for optimization\n"
    else:
        report += "1. Monitor near-critical paths during implementation\n"
        report += "2. Apply standard optimization techniques\n"
    
    report += "\n"
    
    return report


def run_timing_analysis(rtl_code, clock_period_ns=2.0, model_path='xgb_depth_predictor.joblib'):
    """
    Run timing analysis on the provided RTL code.
    
    Parameters:
    -----------
    rtl_code : str
        Verilog RTL code
    clock_period_ns : float
        Clock period in nanoseconds
    model_path : str
        Path to the trained model
        
    Returns:
    --------
    str
        Formatted timing report
    """
    try:
        results_df, module_name = analyze_rtl_module(rtl_code, model_path)

        # Generate and return the timing report
        if not results_df.empty:
            timing_report = generate_timing_report(results_df, module_name, clock_period_ns)
            return timing_report
        else:
            return "No signals found in the RTL code."
    except Exception as e:
        return f"Error during analysis: {str(e)}"


# Example Usage:
if __name__ == "__main__":
    # Assume 'rtl_code' contains your Verilog code and 'xgb_depth_predictor.joblib' is the trained ML model file path
    rtl_code = """module alu(
        input [31:0] a,
        input [31:0] b,
        input [3:0] op,
        output reg [31:0] result,
        output reg overflow
    );
        always @(*) begin
            case(op)
                4'b0000: result = a + b;
                4'b0001: result = a - b;
                4'b0010: result = a & b;
                4'b0011: result = a | b;
                4'b0100: result = a ^ b;
                4'b0101: result = ~a;
                4'b0110: result = a << b[4:0];
                4'b0111: result = a >> b[4:0];
                default: result = 32'h0;
            endcase
            overflow = (op == 4'b0000 && (a[31] == b[31]) && (result[31] != a[31])) || 
                    (op == 4'b0001 && (a[31] != b[31]) && (result[31] != a[31]));
        end
    endmodule"""

    # Run timing analysis with 2.0ns clock period (500MHz)
    timing_report = run_timing_analysis(rtl_code, 2.0)
    print(timing_report)
