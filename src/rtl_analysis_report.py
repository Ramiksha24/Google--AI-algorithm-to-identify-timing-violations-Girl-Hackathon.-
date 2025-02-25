import re
import numpy as np
import pandas as pd
import joblib

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
            report += f"  Setup Slack: {path['setup_slack_ns']:.2f} ns\n"
            report += f"  Fan-in: {path['fanin']}, Fan-out: {path['fanout']}\n"
            report += "\n"
    else:
        report += "No near-critical paths found.\n\n"
    
    # Summary section
    report += "Summary:\n"
    report += "--------\n"
    report += f"Worst Slack: {results_df['setup_slack_ns'].min():.2f} ns\n"
    report += f"Average Slack: {results_df['setup_slack_ns'].mean():.2f} ns\n"
    report += f"Recommended Actions: "
    
    if sum(results_df['setup_slack_ns'] < 0) > 0:
        report += "Design requires optimization to meet timing requirements.\n"
        report += "Consider pipelining or restructuring critical paths.\n"
    else:
        report += "Design meets timing requirements.\n"
    
    return report

# Example usage
if __name__ == "__main__":
    # Example ALU RTL
    alu_rtl = """
    module alu_32bit (
        input wire clk,
        input wire rst,
        input wire [31:0] a,
        input wire [31:0] b,
        input wire [3:0] op,
        output reg [31:0] result,
        output reg overflow,
        output reg zero_flag
    );
        reg [31:0] temp_result;
        wire carry_out;
        
        always @(*) begin
            case(op)
                4'b0000: temp_result = a + b;
                4'b0001: temp_result = a - b;
                4'b0010: temp_result = a & b;
                4'b0011: temp_result = a | b;
                4'b0100: temp_result = a ^ b;
                4'b0101: temp_result = ~a;
                4'b0110: temp_result = a << b[4:0];
                4'b0111: temp_result = a >> b[4:0];
                default: temp_result = 32'h0;
            endcase
        end
        
        always @(posedge clk or posedge rst) begin
            if (rst) begin
                result <= 32'h0;
                overflow <= 1'b0;
                zero_flag <= 1'b0;
            end else begin
                result <= temp_result;
                overflow <= (op == 4'b0000 && (a[31] == b[31]) && (temp_result[31] != a[31])) || 
                           (op == 4'b0001 && (a[31] != b[31]) && (temp_result[31] != a[31]));
                zero_flag <= (temp_result == 32'h0);
            end
        end
    endmodule
    """
    
    # Analyze module
    results = analyze_rtl_module(alu_rtl)
    
    # Print results
    print("RTL Analysis Results:")
    print(results[['signal_name', 'predicted_depth', 'criticality', 'fanin', 'fanout']])
    
    # Generate timing report
    report = generate_timing_report(results, "alu_32bit", clock_period_ns=2.0)
    print("\nTiming Report:")
    print(report)
