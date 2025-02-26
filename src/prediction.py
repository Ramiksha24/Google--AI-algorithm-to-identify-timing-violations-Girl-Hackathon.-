import pandas as pd
import numpy as np
import joblib
import re

def extract_features_from_rtl(rtl_code, signal_name):
    """
    Extract features from RTL code related to a given signal.
    
    Parameters:
    -----------
    rtl_code : str
        Verilog RTL code
    signal_name : str
        Name of the signal to analyze
    
    Returns:
    --------
    dict
        Extracted feature values
    """
    # Check if the signal is an output
    is_output = bool(re.search(rf'output\s+.*{signal_name}', rtl_code))

    # Extract signal width
    match = re.search(rf'(input|output|wire|reg)\s*\[([\d:]+)\]\s*{signal_name}', rtl_code)
    signal_width = eval(match.group(2).replace(':', '-')) + 1 if match else 1

    # Count occurrences of operations related to the signal
    combinational_ops = len(re.findall(rf'{signal_name}\s*=\s*[^\n;]+[+\-*^&|]', rtl_code))
    mux_ops = len(re.findall(rf'{signal_name}\s*=\s*.*\? .+ : .+;', rtl_code))
    if_statements = len(re.findall(rf'if\s*\(.+\)', rtl_code))
    loop_constructs = len(re.findall(r'(for|while|repeat)\s*\(.+\)', rtl_code))
    always_blocks = len(re.findall(r'always\s*@', rtl_code))  # Detect sequential logic
    case_statements = len(re.findall(r'case\s*\(.+\)', rtl_code))
    
    # Fan-in and fan-out (approximation)
    fanin = len(re.findall(rf'input\s+.*\b{signal_name}\b', rtl_code))
    fanout = len(re.findall(rf'output\s+.*\b{signal_name}\b', rtl_code))

    # Count arithmetic operations
    arithmetic_ops = len(re.findall(rf'{signal_name}\s*=\s*.*[+\-*\/]', rtl_code))

    # Compute module complexity (dummy metric: number of lines)
    module_complexity = len(rtl_code.split('\n'))

    return {
        'signal_width': signal_width,
        'is_output': int(is_output),
        'is_registered': 0,  # Placeholder for flip-flop detection
        'combinational_ops': combinational_ops,
        'mux_ops': mux_ops,
        'if_statements': if_statements,
        'loop_constructs': loop_constructs,
        'always_blocks': always_blocks,
        'case_statements': case_statements,
        'fanin': fanin,
        'fanout': fanout,
        'arithmetic_ops': arithmetic_ops,
        'module_complexity': module_complexity
    }


def determine_module_type(rtl_code):
    """Determine the type of module based on RTL code."""
    if re.search(r'add|sum|carry', rtl_code, re.IGNORECASE):
        return 'Adder'
    elif re.search(r'mult|product', rtl_code, re.IGNORECASE):
        return 'Multiplier'
    elif re.search(r'alu|arithmetic', rtl_code, re.IGNORECASE):
        return 'ALU'
    elif re.search(r'fifo|queue|buffer', rtl_code, re.IGNORECASE):
        return 'FIFO'
    elif re.search(r'fsm|state|case', rtl_code, re.IGNORECASE):  # Improved FSM detection
        return 'FSM'
    elif re.search(r'count|increment', rtl_code, re.IGNORECASE):
        return 'Counter'
    elif re.search(r'mem|ram|rom', rtl_code, re.IGNORECASE):
        return 'MemoryController'
    elif re.search(r'reg|flip\-flop', rtl_code, re.IGNORECASE):
        return 'ShiftRegister'
    elif re.search(r'decode', rtl_code, re.IGNORECASE):
        return 'Decoder'
    else:
        return 'Other'

def predict_combinational_depth(rtl_code, signal_name, model_path='xgb_depth_predictor.joblib'):
    """
    Predict the combinational depth of a signal in an RTL module.
    """
    model = joblib.load(model_path)
    
    features = extract_features_from_rtl(rtl_code, signal_name)
    
    features['module_type'] = determine_module_type(rtl_code)
    features['technology_node_nm'] = 28  # Default value
    features['clock_frequency_mhz'] = 500  # Default value
    features['optimization_level'] = 2  # Default value
    
    features_df = pd.DataFrame([features])
    
    return model.predict(features_df)[0]

# Example RTL code for ALU
alu_rtl = """
module alu_32bit (
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
endmodule
"""

# Predict combinational depth
result_depth = predict_combinational_depth(alu_rtl, 'result')
overflow_depth = predict_combinational_depth(alu_rtl, 'overflow')

print(f"Predicted combinational depth for 'result' signal: {result_depth:.2f}")
print(f"Predicted combinational depth for 'overflow' signal: {overflow_depth:.2f}")
