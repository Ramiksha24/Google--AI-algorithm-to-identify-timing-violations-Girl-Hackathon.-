import re
import os
import sys
import pandas as pd
import numpy as np
from collections import defaultdict, Counter

class RTLTimingFeatureExtractor:
    def __init__(self):
        self.features = {}
        self.module_name = ""
        self.clock_frequency = 0
        self.timing_paths = []
        self.rtl_features = {}
        
    def extract_all_features(self, verilog_file, timing_report_file):
        """Extract all features from Verilog and timing report files"""
        # Extract features from RTL
        self.extract_rtl_features(verilog_file)
        
        # Extract features from timing report
        self.extract_timing_features(timing_report_file)
        
        # Combine all features
        self.combine_features()
        
        return self.features
    
    def extract_rtl_features(self, verilog_file):
        """Extract features from Verilog RTL file"""
        try:
            with open(verilog_file, 'r') as f:
                verilog_content = f.read()
                
            # Extract module name
            module_match = re.search(r'module\s+(\w+)', verilog_content)
            if module_match:
                self.module_name = module_match.group(1)
                self.rtl_features['module_name'] = self.module_name
            
            # Extract input and output ports
            input_ports = re.findall(r'input\s+(?:wire\s+)?(?:\[\d+:\d+\]\s+)?(\w+)', verilog_content)
            output_ports = re.findall(r'output\s+(?:wire\s+|reg\s+)?(?:\[\d+:\d+\]\s+)?(\w+)', verilog_content)
            
            self.rtl_features['input_count'] = len(input_ports)
            self.rtl_features['output_count'] = len(output_ports)
            self.rtl_features['io_ratio'] = len(input_ports) / max(1, len(output_ports))
            
            # Extract register declarations
            reg_declarations = re.findall(r'reg\s+(?:\[\d+:\d+\]\s+)?(\w+)', verilog_content)
            self.rtl_features['register_count'] = len(reg_declarations)
            
            # Extract wire declarations
            wire_declarations = re.findall(r'wire\s+(?:\[\d+:\d+\]\s+)?(\w+)', verilog_content)
            self.rtl_features['wire_count'] = len(wire_declarations)
            
            # Extract always blocks
            always_blocks = re.findall(r'always\s+@\s*\([^)]*\)', verilog_content)
            self.rtl_features['always_block_count'] = len(always_blocks)
            
            # Count sequential logic (clocked always blocks)
            sequential_always = len(re.findall(r'always\s+@\s*\(\s*(?:posedge|negedge)\s+\w+', verilog_content))
            self.rtl_features['sequential_block_count'] = sequential_always
            
            # Count combinational logic (non-clocked always blocks)
            self.rtl_features['combinational_block_count'] = len(always_blocks) - sequential_always
            
            # Extract bitwidth information
            bitwidth_matches = re.findall(r'\[(\d+):(\d+)\]', verilog_content)
            if bitwidth_matches:
                bitwidths = [abs(int(high) - int(low)) + 1 for high, low in bitwidth_matches]
                self.rtl_features['max_bitwidth'] = max(bitwidths) if bitwidths else 0
                self.rtl_features['avg_bitwidth'] = sum(bitwidths) / len(bitwidths) if bitwidths else 0
            else:
                self.rtl_features['max_bitwidth'] = 1
                self.rtl_features['avg_bitwidth'] = 1
            
            # Count operators
            self.rtl_features['addition_ops'] = len(re.findall(r'[^<>!=]=?\+', verilog_content))
            self.rtl_features['subtraction_ops'] = len(re.findall(r'[^<](?<!=)-', verilog_content))
            self.rtl_features['multiplication_ops'] = len(re.findall(r'\*', verilog_content))
            self.rtl_features['division_ops'] = len(re.findall(r'/', verilog_content))
            self.rtl_features['logical_ops'] = len(re.findall(r'&&|\|\||!(?!=)', verilog_content))
            self.rtl_features['bitwise_ops'] = len(re.findall(r'&(?!&)|\|(?!\|)|\^|~', verilog_content))
            self.rtl_features['shift_ops'] = len(re.findall(r'<<|>>', verilog_content))
            self.rtl_features['comparison_ops'] = len(re.findall(r'==|!=|<=|>=|<(?!<)|(?<!>)>', verilog_content))
            
            # Count conditional constructs
            self.rtl_features['if_count'] = len(re.findall(r'\bif\s*\(', verilog_content))
            self.rtl_features['else_count'] = len(re.findall(r'\belse\b', verilog_content))
            self.rtl_features['case_count'] = len(re.findall(r'\bcase\b', verilog_content))
            
            # Count procedural blocks
            self.rtl_features['begin_end_pairs'] = len(re.findall(r'\bbegin\b', verilog_content))
            
            # Calculate basic complexity metrics
            self.rtl_features['cyclomatic_complexity'] = (
                self.rtl_features['if_count'] + 
                self.rtl_features['case_count'] + 
                1
            )
            
            # Count signal assignments
            blocking_assignments = len(re.findall(r'=(?!=)', verilog_content))
            non_blocking_assignments = len(re.findall(r'<=(?!<)', verilog_content))
            self.rtl_features['blocking_assignments'] = blocking_assignments
            self.rtl_features['non_blocking_assignments'] = non_blocking_assignments
            self.rtl_features['total_assignments'] = blocking_assignments + non_blocking_assignments
            
            # Analyze signal fanout (based on variable name occurrence after declaration)
            signal_counts = {}
            all_signals = reg_declarations + wire_declarations
            for signal in all_signals:
                # Count signal occurrences excluding the declaration
                signal_pattern = r'\b' + re.escape(signal) + r'\b'
                occurrences = len(re.findall(signal_pattern, verilog_content)) - 1  # -1 to exclude declaration
                signal_counts[signal] = max(0, occurrences)
            
            if signal_counts:
                self.rtl_features['max_fanout'] = max(signal_counts.values()) if signal_counts else 0
                self.rtl_features['avg_fanout'] = sum(signal_counts.values()) / len(signal_counts) if signal_counts else 0
            else:
                self.rtl_features['max_fanout'] = 0
                self.rtl_features['avg_fanout'] = 0
            
            # Calculate total lines of code and code density
            loc = len(verilog_content.split('\n'))
            self.rtl_features['loc'] = loc
            self.rtl_features['operator_density'] = (
                self.rtl_features['addition_ops'] + 
                self.rtl_features['subtraction_ops'] + 
                self.rtl_features['multiplication_ops'] + 
                self.rtl_features['division_ops'] + 
                self.rtl_features['logical_ops'] + 
                self.rtl_features['bitwise_ops'] + 
                self.rtl_features['shift_ops'] + 
                self.rtl_features['comparison_ops']
            ) / max(1, loc)
            
            # Count instances (module instantiations)
            instance_matches = re.findall(r'(\w+)\s+(?:\w+\s+)?(\w+)\s*\(', verilog_content)
            instances = [match[0] for match in instance_matches if match[0] != 'module' and match[0] != 'function']
            self.rtl_features['instance_count'] = len(instances)
            
            # Count unique instance types
            self.rtl_features['unique_instance_types'] = len(set(instances))
            
            # Calculate hierarchy depth heuristic (based on instance nesting)
            hierarchy_depth = 1  # At least the current module
            if instances:
                # Estimate depth based on indentation patterns
                indentation_patterns = re.findall(r'(\s+)\w+\s+\w+\s*\(', verilog_content)
                if indentation_patterns:
                    max_indent = max(len(indent) for indent in indentation_patterns)
                    estimated_depth = max_indent // 2 + 1  # Assuming 2 spaces per level
                    hierarchy_depth = max(hierarchy_depth, estimated_depth)
            
            self.rtl_features['hierarchy_depth'] = hierarchy_depth
            
            print(f"Successfully extracted {len(self.rtl_features)} RTL features from {verilog_file}")
            
        except Exception as e:
            print(f"Error extracting RTL features: {e}")
            self.rtl_features = {
                'module_name': 'unknown',
                'error': str(e)
            }
    
    def extract_timing_features(self, timing_report_file):
        """Extract features from timing report file"""
        try:
            with open(timing_report_file, 'r') as f:
                timing_content = f.read()
            
            # Extract module name from report
            module_match = re.search(r'Module:\s+(\w+)', timing_content)
            if module_match:
                report_module = module_match.group(1)
                if self.module_name and report_module != self.module_name:
                    print(f"Warning: Module name mismatch between RTL ({self.module_name}) and timing report ({report_module})")
                self.module_name = report_module
            
            # Extract clock frequency (if present)
            clock_match = re.search(r'Clock:\s+\(R\)\s+(\w+)\s+Period:\s+([\d\.]+)', timing_content)
            if clock_match:
                clock_name = clock_match.group(1)
                clock_period = float(clock_match.group(2))
                self.clock_frequency = 1000 / clock_period if clock_period else 0  # MHz
            
            # Parse path details
            path_blocks = re.findall(r'Path \d+:(.*?)(?=Path \d+:|$)', timing_content, re.DOTALL)
            
            timing_features = {}
            all_paths = []
            
            # Process each timing path
            for path_idx, path_block in enumerate(path_blocks):
                path_features = {}
                
                # Extract slack
                slack_match = re.search(r'Slack:\s*=\s*([-\d\.]+)', path_block)
                if slack_match:
                    path_features['slack'] = float(slack_match.group(1))
                
                # Check if violated
                violated_match = re.search(r'VIOLATED\s+\(([-\d\.]+)\s+ps\)', path_block)
                path_features['is_violated'] = 1 if violated_match else 0
                if violated_match:
                    path_features['violation_amount'] = float(violated_match.group(1))
                else:
                    path_features['violation_amount'] = 0
                
                # Extract required time
                required_match = re.search(r'Required Time:\s*=\s*([-\d\.]+)', path_block)
                if required_match:
                    path_features['required_time'] = float(required_match.group(1))
                
                # Extract arrival time
                arrival_match = re.search(r'Arrival:\s*=\s*([-\d\.]+)', path_block)
                if arrival_match:
                    path_features['arrival_time'] = float(arrival_match.group(1))
                
                # Extract data path delay
                data_path_match = re.search(r'Data Path:\s*-\s*([-\d\.]+)', path_block)
                if data_path_match:
                    path_features['data_path_delay'] = float(data_path_match.group(1))
                
                # Extract startpoint and endpoint
                startpoint_match = re.search(r'Startpoint:\s+\([R]\)\s+([^/\s]+)', path_block)
                endpoint_match = re.search(r'Endpoint:\s+\([R]\)\s+([^/\s]+)', path_block)
                
                if startpoint_match:
                    path_features['startpoint'] = startpoint_match.group(1)
                if endpoint_match:
                    path_features['endpoint'] = endpoint_match.group(1)
                
                # Path type (reg-to-reg, reg-to-output, input-to-reg, etc.)
                if startpoint_match and endpoint_match:
                    start = startpoint_match.group(1)
                    end = endpoint_match.group(1)
                    
                    if "_reg" in start and "_reg" in end:
                        path_features['path_type'] = "reg_to_reg"
                    elif "_reg" in start and "_reg" not in end:
                        path_features['path_type'] = "reg_to_output"
                    elif "_reg" not in start and "_reg" in end:
                        path_features['path_type'] = "input_to_reg"
                    else:
                        path_features['path_type'] = "comb_logic"
                
                # Extract detailed path information
                path_details = []
                path_detail_lines = re.findall(r'[\w/]+\s+[-]\s+[-\w>]+\s+[RF]\s+\w+\s+\d+\s+[\d\.]+\s+\d+\s+\d+\s+\d+', path_block)
                
                cell_types = []
                transitions = []
                delays = []
                loads = []
                
                for detail in path_detail_lines:
                    fields = detail.split()
                    if len(fields) >= 9:
                        instance = fields[0]
                        cell_type = fields[4]
                        fanout = int(fields[5]) if fields[5].isdigit() else 0
                        load = float(fields[6]) if is_number(fields[6]) else 0
                        trans = float(fields[7]) if is_number(fields[7]) else 0
                        delay = float(fields[8]) if is_number(fields[8]) else 0
                        
                        cell_types.append(cell_type)
                        transitions.append(trans)
                        delays.append(delay)
                        loads.append(load)
                        
                        path_details.append({
                            'instance': instance,
                            'cell_type': cell_type,
                            'fanout': fanout,
                            'load': load,
                            'transition': trans,
                            'delay': delay
                        })
                
                # Calculate logic depth (number of cells in path)
                path_features['logic_depth'] = len(path_details)
                
                # Path delay and transition metrics
                if delays:
                    path_features['total_cell_delay'] = sum(delays)
                    path_features['max_cell_delay'] = max(delays)
                    path_features['avg_cell_delay'] = sum(delays) / len(delays)
                
                if transitions:
                    path_features['max_transition'] = max(transitions)
                    path_features['avg_transition'] = sum(transitions) / len(transitions)
                
                if loads:
                    path_features['max_load'] = max(loads)
                    path_features['avg_load'] = sum(loads) / len(loads)
                
                # Cell type distribution
                cell_type_counts = Counter(cell_types)
                for cell_type, count in cell_type_counts.items():
                    path_features[f'cell_type_{cell_type}'] = count
                
                path_features['unique_cell_types'] = len(cell_type_counts)
                
                # Store path details
                path_features['path_details'] = path_details
                all_paths.append(path_features)
            
            # Get worst (most critical) path
            if all_paths:
                worst_path = min(all_paths, key=lambda x: x.get('slack', 0))
                
                # Extract timing features from worst path
                timing_features['worst_slack'] = worst_path.get('slack', 0)
                timing_features['worst_logic_depth'] = worst_path.get('logic_depth', 0)
                timing_features['worst_data_path_delay'] = worst_path.get('data_path_delay', 0)
                timing_features['worst_path_type'] = worst_path.get('path_type', 'unknown')
                
                # Average metrics across all paths
                timing_features['avg_logic_depth'] = sum(p.get('logic_depth', 0) for p in all_paths) / len(all_paths)
                timing_features['avg_slack'] = sum(p.get('slack', 0) for p in all_paths) / len(all_paths)
                timing_features['violation_count'] = sum(1 for p in all_paths if p.get('is_violated', 0) == 1)
                timing_features['violation_percentage'] = timing_features['violation_count'] / len(all_paths)
                
                # Critical path cell distribution
                if 'path_details' in worst_path:
                    cell_types = [d['cell_type'] for d in worst_path['path_details']]
                    cell_counts = Counter(cell_types)
                    for cell_type, count in cell_counts.items():
                        timing_features[f'critical_path_cell_{cell_type}'] = count
                
                # Clock frequency
                timing_features['clock_frequency_mhz'] = self.clock_frequency
            
            # Extract area information
            area_match = re.search(r'Cell Count\s+Cell Area\s+Net Area\s+Total Area(.*?)(?:--|$)', timing_content, re.DOTALL)
            if area_match:
                area_line = area_match.group(1).strip()
                area_fields = re.split(r'\s+', area_line)
                
                if len(area_fields) >= 4:
                    try:
                        timing_features['cell_count'] = int(area_fields[-4])
                        timing_features['cell_area'] = float(area_fields[-3])
                        timing_features['net_area'] = float(area_fields[-2])
                        timing_features['total_area'] = float(area_fields[-1])
                    except (ValueError, IndexError):
                        pass
            
            # Store all path information
            self.timing_paths = all_paths
            self.timing_features = timing_features
            
            print(f"Successfully extracted {len(self.timing_features)} timing features from {timing_report_file}")
            
        except Exception as e:
            print(f"Error extracting timing features: {e}")
            self.timing_features = {
                'error': str(e)
            }
    
    def combine_features(self):
        """Combine RTL and timing features into one feature set"""
        # Base features
        self.features = {
            'module_name': self.module_name,
            'clock_frequency_mhz': self.clock_frequency
        }
        
        # Add RTL features
        for key, value in self.rtl_features.items():
            self.features[f'rtl_{key}'] = value
        
        # Add timing features
        for key, value in self.timing_features.items():
            self.features[f'timing_{key}'] = value
        
        # Add derived combinational depth features
        if hasattr(self, 'timing_paths') and self.timing_paths:
            # Use worst path logic depth as primary combinational depth metric
            worst_path = min(self.timing_paths, key=lambda x: x.get('slack', 0))
            self.features['combinational_depth'] = worst_path.get('logic_depth', 0)
            
            # Additional combinational depth metrics
            reg_to_reg_paths = [p for p in self.timing_paths if p.get('path_type') == 'reg_to_reg']
            if reg_to_reg_paths:
                self.features['reg_to_reg_depth'] = max(p.get('logic_depth', 0) for p in reg_to_reg_paths)
            
            # Timing violation prediction features
            self.features['timing_violation_risk'] = 1 if any(p.get('is_violated', 0) == 1 for p in self.timing_paths) else 0
            
            violation_paths = [p for p in self.timing_paths if p.get('is_violated', 0) == 1]
            if violation_paths:
                self.features['worst_violation'] = min(p.get('slack', 0) for p in violation_paths)
            else:
                self.features['worst_violation'] = 0
            
            # Compute additional metrics that might be useful
            self.features['timing_margin'] = min(p.get('slack', 0) for p in self.timing_paths)
            
            # Logic depth distribution
            depths = [p.get('logic_depth', 0) for p in self.timing_paths]
            self.features['max_logic_depth'] = max(depths)
            self.features['min_logic_depth'] = min(depths)
            self.features['std_logic_depth'] = np.std(depths) if depths else 0
            
            # Logic cone size heuristic
            if 'rtl_wire_count' in self.features and 'rtl_register_count' in self.features:
                self.features['logic_cone_size_heuristic'] = (
                    self.features['rtl_wire_count'] + 
                    self.features['rtl_register_count']
                ) / max(1, len(self.timing_paths))
        
        print(f"Combined all features: {len(self.features)} total features extracted")
        return self.features
    
    def save_features_to_csv(self, output_file):
        """Save extracted features to a CSV file"""
        try:
            # Convert features to DataFrame
            df = pd.DataFrame([self.features])
            
            # Save to CSV
            df.to_csv(output_file, index=False)
            print(f"Features saved to {output_file}")
            return True
        except Exception as e:
            print(f"Error saving features to CSV: {e}")
            return False
            
    def print_key_features(self):
        """Print key features that are most relevant to combinational depth prediction"""
        key_features = [
            'module_name',
            'clock_frequency_mhz',
            'combinational_depth',
            'timing_violation_risk',
            'worst_violation',
            'timing_margin',
            'max_logic_depth',
            'rtl_combinational_block_count',
            'rtl_sequential_block_count',
            'rtl_cyclomatic_complexity',
            'rtl_max_fanout',
            'rtl_addition_ops',
            'rtl_multiplication_ops',
            'rtl_bitwise_ops',
            'rtl_if_count',
            'rtl_case_count',
            'timing_worst_logic_depth',
            'timing_worst_slack',
            'timing_worst_data_path_delay',
            'timing_avg_logic_depth',
            'timing_violation_count'
        ]
        
        print("\n--- Key Features for Combinational Depth Prediction ---")
        for feature in key_features:
            if feature in self.features:
                value = self.features[feature]
                if isinstance(value, float):
                    formatted_value = f"{value:.3f}"
                else:
                    formatted_value = str(value)
                print(f"{feature}: {formatted_value}")
            else:
                print(f"{feature}: Not available")

def is_number(s):
    """Check if string is a number"""
    try:
        float(s)
        return True
    except ValueError:
        return False

def main():
    if len(sys.argv) < 3:
        print("Usage: python extract_features.py <verilog_file> <timing_report_file> [output_csv]")
        sys.exit(1)
    
    verilog_file = sys.argv[1]
    timing_report_file = sys.argv[2]
    
    output_file = sys.argv[3] if len(sys.argv) > 3 else "Cadence_training2.csv"
    
    extractor = RTLTimingFeatureExtractor()
    extractor.extract_all_features(verilog_file, timing_report_file)
    extractor.print_key_features()
    extractor.save_features_to_csv(output_file)

if __name__ == "__main__":
    main()
