import re
import numpy as np

def extract_features_from_rtl(rtl_code, signal_name):
    """
    Extract features from RTL code for a specific signal.
    
    Parameters:
    -----------
    rtl_code : str
        Verilog/VHDL code to analyze
    signal_name : str
        Name of the signal to analyze
    
    Returns:
    --------
    dict
        Dictionary of extracted features
    """
    features = {}
    
    # Basic signal characteristics
    signal_decl = re.search(rf'(?:output|input|wire|reg)\s+(?:\[\d+:\d+\])?\s*{signal_name}\b', rtl_code)
    if signal_decl:
        features['is_output'] = 1 if 'output' in signal_decl.group(0) else 0
        features['is_registered'] = 1 if 'reg' in signal_decl.group(0) else 0
    else:
        features['is_output'] = 0
        features['is_registered'] = 0
    
    # Signal width
    width_match = re.search(rf'(?:output|input|wire|reg)\s+\[(\d+):(\d+)\]\s*{signal_name}\b', rtl_code)
    if width_match:
        high, low = int(width_match.group(1)), int(width_match.group(2))
        features['signal_width'] = abs(high - low) + 1
    else:
        features['signal_width'] = 1
    
    # Estimate fan-in (number of signals affecting this signal)
    fanin = 0
    for match in re.finditer(rf'{signal_name}\s*(?:<=|=)\s*([^;]+);', rtl_code):
        expr = match.group(1)
        # Find all signal names in the expression
        signals = re.findall(r'\b[a-zA-Z]\w*\b', expr)
        fanin += len(set(signals))
    
    features['fanin'] = max(1, fanin)
    
    # Estimate fan-out (how many other signals this signal affects)
    fanout = len(re.findall(rf'\b{signal_name}\b[^=]*?(?:<=|=)', rtl_code))
    features['fanout'] = max(1, fanout)
    
    # Count logic operations
    for match in re.finditer(rf'{signal_name}\s*(?:<=|=)\s*([^;]+);', rtl_code):
        expr = match.group(1)
        features['combinational_ops'] = len(re.findall(r'[&|^~]', expr))
        features['arithmetic_ops'] = len(re.findall(r'[+\-*/]', expr))
        features['mux_ops'] = len(re.findall(r'\?', expr))
    
    # Module complexity features
    features['always_blocks'] = len(re.findall(r'always\s+@', rtl_code))
    features['case_statements'] = len(re.findall(r'case\s*\(', rtl_code))
    features['if_statements'] = len(re.findall(r'if\s*\(', rtl_code))
    features['loop_constructs'] = len(re.findall(r'for\s*\(', rtl_code))
    
    # If any features are missing, set defaults
    default_features = {
        'combinational_ops': 0,
        'arithmetic_ops': 0,
        'mux_ops': 0
    }
    
    for key, value in default_features.items():
        if key not in features:
            features[key] = value
    
    # Module complexity score (1-10 scale)
    complexity_score = (
        1.0 * features['always_blocks'] + 
        0.5 * features['case_statements'] + 
        0.3 * features['if_statements'] + 
        0.2 * features['loop_constructs'] +
        0.1 * features['combinational_ops'] +
        0.2 * features['arithmetic_ops']
    )
    
    features['module_complexity'] = min(10, max(1, int(complexity_score)))
    
    return features

def batch_extract_features(rtl_code):
    """
    Extract features for multiple signals in the RTL code.
    
    Parameters:
    -----------
    rtl_code : str
        Verilog/VHDL code to analyze
    
    Returns:
    --------
    list
        List of feature dictionaries for each signal
    """
    # Find all signal declarations
    signal_patterns = [
        r'output\s+(?:\[\d+:\d+\])?\s*(\w+)',
        r'input\s+(?:\[\d+:\d+\])?\s*(\w+)',
        r'wire\s+(?:\[\d+:\d+\])?\s*(\w+)',
        r'reg\s+(?:\[\d+:\d+\])?\s*(\w+)'
    ]
    
    signals = []
    for pattern in signal_patterns:
        signals.extend(re.findall(pattern, rtl_code))
    
    # Remove duplicates
    signals = list(set(signals))
    
    # Extract features for each signal
    features_list = []
    for signal in signals:
        signal_features = extract_features_from_rtl(rtl_code, signal)
        signal_features['signal_name'] = signal
        features_list.append(signal_features)
    
    return features_list

# Example usage
if __name__ == '__main__':
    # Sample RTL code for testing
    sample_rtl = '''
    module example_module (
        input wire clk,
        input wire reset,
        input [7:0] data_in,
        output reg [15:0] result
    );
        reg [3:0] counter;
        wire intermediate;
        
        always @(posedge clk) begin
            if (reset) begin
                result <= 0;
                counter <= 0;
            end else begin
                result <= data_in * 2 + counter;
                counter <= counter + 1;
            end
        end
        
        assign intermediate = data_in & 4'hF;
    endmodule
    '''
    
    # Extract features for all signals
    extracted_features = batch_extract_features(sample_rtl)
    
    # Print extracted features
    for features in extracted_features:
        print(f"Signal: {features.get('signal_name', 'Unknown')}")
        for key, value in features.items():
            print(f"  {key}: {value}")
        print()
