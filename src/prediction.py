import pandas as pd
import numpy as np
import networkx as nx
import re
from collections import defaultdict
import joblib
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import argparse
import sys
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import colorama
from colorama import Fore, Back, Style
from tabulate import tabulate
import json
import datetime

# Initialize colorama for cross-platform colored terminal output
colorama.init()

class RTLTimingAnalyzer:
    def __init__(self, model_path=None):
        """Initialize the RTL Timing Analyzer with models and configuration"""
        self.banner()
        print(f"{Fore.CYAN}Initializing RTL Timing Analyzer...{Style.RESET_ALL}")
        
        # Load pre-trained models if available, otherwise use defaults
        try:
            self.timing_violation_model = joblib.load('timing_violation_model.pkl')
            self.combinational_depth_model = joblib.load('combinational_depth_model.pkl')
            self.module_encoder = joblib.load('module_name_encoder.pkl')
            self.risk_encoder = joblib.load('risk_encoder.pkl')
            self.timing_feature_names = joblib.load('timing_feature_names.pkl')
            self.depth_feature_names = joblib.load('depth_feature_names.pkl')
            print(f"{Fore.GREEN}✓ Models loaded successfully{Style.RESET_ALL}")
        except:
            print(f"{Fore.YELLOW}! Using default models - for demonstration purposes only{Style.RESET_ALL}")
            # Create simplified default models for demonstration
            self.timing_violation_model = self._create_default_model('classification')
            self.combinational_depth_model = self._create_default_model('regression')
            self.timing_feature_names = []
            self.depth_feature_names = []
    
    def banner(self):
        """Display a fancy banner for the application"""
        banner = f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║  {Fore.YELLOW}██████╗ ████████╗██╗     ██╗███╗   ██╗███████╗██╗ ██████╗ ██╗  ██╗████████╗{Fore.CYAN} ║
║  {Fore.YELLOW}██╔══██╗╚══██╔══╝██║     ██║████╗  ██║██╔════╝██║██╔════╝ ██║  ██║╚══██╔══╝{Fore.CYAN} ║
║  {Fore.YELLOW}██████╔╝   ██║   ██║     ██║██╔██╗ ██║███████╗██║██║  ███╗███████║   ██║   {Fore.CYAN} ║
║  {Fore.YELLOW}██╔══██╗   ██║   ██║     ██║██║╚██╗██║╚════██║██║██║   ██║██╔══██║   ██║   {Fore.CYAN} ║
║  {Fore.YELLOW}██║  ██║   ██║   ███████╗██║██║ ╚████║███████║██║╚██████╔╝██║  ██║   ██║   {Fore.CYAN} ║
║  {Fore.YELLOW}╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝╚═╝  ╚═══╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝   {Fore.CYAN} ║
║                                                              ║
║  {Fore.GREEN}AI-Powered Timing & Design Analyzer v2.0{Fore.CYAN}                    ║
║  {Fore.WHITE}© 2025 Advanced RTL Design Solutions{Fore.CYAN}                        ║
╚══════════════════════════════════════════════════════════════╝{Style.RESET_ALL}
"""
        print(banner)
    
    def _create_default_model(self, model_type):
        """Create a simple default model for demonstration purposes"""
        if model_type == 'classification':
            return Pipeline([
                ('scaler', StandardScaler()),
                ('xgb', xgb.XGBClassifier(n_estimators=50, max_depth=3))
            ])
        else:
            return Pipeline([
                ('scaler', StandardScaler()),
                ('xgb', xgb.XGBRegressor(n_estimators=50, max_depth=3))
            ])

    def _extract_rtl_features(self, rtl_code, user_params=None):
        """
        Extract features from RTL code with user-specified parameters
        
        Args:
            rtl_code (str): The RTL code to analyze
            user_params (dict): User-specified parameters including clock_frequency, technology, etc.
        """
        features = {}
        
        # Extract module name
        module_match = re.search(r'module\s+(\w+)', rtl_code)
        if module_match:
            features['module_name'] = module_match.group(1)
        else:
            features['module_name'] = "unknown_module"
        
        # Count gates and structures
        features['cell_count'] = len(re.findall(r'(assign|always|if|else|case)', rtl_code))
        features['combinational_logic'] = len(re.findall(r'assign', rtl_code))
        features['arithmetic_ops'] = len(re.findall(r'(\+|-|\*|\/)', rtl_code))
        features['multiplexers'] = len(re.findall(r'(\?|case|if\s*\(.*\)\s*)', rtl_code))
        
        # Estimate logic types percentages based on operators found
        total_logic = features['combinational_logic'] + 1  # Avoid division by zero
        
        # Count different gate types
        and_gates = len(re.findall(r'(&|\band\b)', rtl_code))
        or_gates = len(re.findall(r'(\||\bor\b)', rtl_code))
        not_gates = len(re.findall(r'(~|!|\bnot\b)', rtl_code))
        xor_gates = len(re.findall(r'(\^|\bxor\b)', rtl_code))
        buffers = len(re.findall(r'(buf|buffer)', rtl_code))
        
        # Calculate percentages
        total_gates = and_gates + or_gates + not_gates + xor_gates + buffers
        total_gates = max(total_gates, 1)  # Avoid division by zero
        
        features['logic_and_percent'] = (and_gates / total_gates) * 100
        features['logic_or_percent'] = (or_gates / total_gates) * 100
        features['logic_not_percent'] = (not_gates / total_gates) * 100
        features['logic_xor_percent'] = (xor_gates / total_gates) * 100
        features['logic_buffer_percent'] = (buffers / total_gates) * 100
        
        # Sequential elements
        sequential_elements = len(re.findall(r'(always\s*@\s*\(posedge|reg\s|flop|ff|dff)', rtl_code))
        features['logic_sequential_percent'] = (sequential_elements / (total_gates + sequential_elements)) * 100
        
        # Estimate maximum logic depth
        lines = rtl_code.split('\n')
        max_indent = 0
        for line in lines:
            indent = len(line) - len(line.lstrip())
            max_indent = max(max_indent, indent)
        
        # Estimate max logic depth based on indentation and assign statements
        features['max_logic_depth'] = max(1, max_indent // 2)
        
        # Estimate combinational depth from RTL
        # More complex RTL would need graph-based analysis
        features['rtl_max_fanout'] = 4 + features['arithmetic_ops'] // 2  # Simplified estimate
        
        # Use user-provided parameters if available
        if user_params:
            features['clock_frequency_mhz'] = user_params.get('clock_frequency_mhz', 500)
            features['technology'] = user_params.get('technology', '45')
            
            # Adjust delay estimates based on technology node
            tech_factor = {
                '7': 0.4,
                '10': 0.5,
                '14': 0.6,
                '22': 0.7,
                '28': 0.8,
                '45': 1.0,
                '65': 1.3,
                '90': 1.6,
                '130': 2.0,
                '180': 2.5
            }.get(features['technology'], 1.0)
            
            # Adjust required time based on clock frequency
            clock_period_ps = 1000000 / features['clock_frequency_mhz']
            features['required_time_ps'] = clock_period_ps * 0.8  # 80% of clock period as required time
            
            # Compute data path delay with technology scaling
            features['data_path_delay_ps'] = (800 + 100 * features['max_logic_depth']) * tech_factor
            
            # Custom parameters
            if 'custom_delay_factor' in user_params:
                features['data_path_delay_ps'] *= user_params['custom_delay_factor']
                
            if 'custom_slack_margin' in user_params:
                features['required_time_ps'] *= (1 - user_params['custom_slack_margin'])
        else:
            # Default values if user parameters not provided
            features['clock_frequency_mhz'] = 500  # Assuming 500MHz default clock
            features['data_path_delay_ps'] = 1000 + 100 * features['max_logic_depth']  # Simple model: base + depth*factor
            features['required_time_ps'] = 2000  # Assuming 2000ps required time
            features['technology'] = '45'  # Default 45nm
            
        # Calculate slack
        features['slack_ps'] = features['required_time_ps'] - features['data_path_delay_ps']
        features['output_delay_ps'] = features['data_path_delay_ps'] - 100  # Simplified
        
        # Risk assessment based on slack
        if features['slack_ps'] > 200:
            features['timing_violation_risk'] = 'Low'
        elif features['slack_ps'] > 0:
            features['timing_violation_risk'] = 'Medium'
        else:
            features['timing_violation_risk'] = 'High'
            
        # For timing analysis
        features['timing_worst_logic_depth'] = features['max_logic_depth']
        features['timing_worst_slack'] = features['slack_ps']
        features['timing_worst_data_path_delay'] = features['data_path_delay_ps']
        features['timing_avg_logic_depth'] = features['max_logic_depth'] * 0.7  # Simplified
        
        # RTL block counts
        features['rtl_combinational_block_count'] = features['combinational_logic']
        features['rtl_sequential_block_count'] = sequential_elements
        
        return features

    def _identify_patterns(self, rtl_code):
        """
        Identify repeated structural patterns in RTL code
        """
        patterns = defaultdict(int)
        
        # Look for common design patterns
        if re.search(r'module\s+.*multiplier|(\*)', rtl_code, re.IGNORECASE):
            patterns['multiplier'] = rtl_code.count('*')
        
        if re.search(r'module\s+.*adder|(\+)', rtl_code, re.IGNORECASE):
            patterns['adder'] = rtl_code.count('+')
        
        if re.search(r'module\s+.*alu', rtl_code, re.IGNORECASE):
            patterns['alu'] = 1
            
        if re.search(r'case\s*\(.*\)', rtl_code, re.IGNORECASE):
            patterns['mux'] = len(re.findall(r'case\s*\(', rtl_code))
            
        if re.search(r'always\s*@\s*\(posedge', rtl_code, re.IGNORECASE):
            patterns['sequential_block'] = len(re.findall(r'always\s*@\s*\(posedge', rtl_code))
            
        # Detect pipeline stages
        pipeline_stages = len(re.findall(r'always\s*@\s*\(posedge.*\)', rtl_code))
        if pipeline_stages > 2:
            patterns['pipeline'] = pipeline_stages
            
        # Detect memory structures
        if re.search(r'(memory|ram|rom|register\s+file)', rtl_code, re.IGNORECASE):
            patterns['memory'] = 1
            
        # Detect finite state machines
        if re.search(r'(state|next_state|current_state)', rtl_code, re.IGNORECASE):
            state_count = len(set(re.findall(r'(state\s*=\s*\w+|state\s*<=\s*\w+)', rtl_code)))
            if state_count > 1:
                patterns['fsm'] = state_count
                
        # Detect clock domain crossing structures
        if re.search(r'(synchronizer|async|metastability)', rtl_code, re.IGNORECASE):
            patterns['cdc'] = 1
            
        # Detect data path vs control path
        control_signals = len(re.findall(r'(enable|valid|ready)', rtl_code)) 
        if control_signals > 2:
            patterns['control_path'] = control_signals
            
        return patterns

    def _create_graph_representation(self, rtl_code):
        """
        Create a graph representation of the RTL design for analysis
        This is a simplified representation - a real implementation would use
        a proper RTL parser and build a complete netlist graph
        """
        G = nx.DiGraph()
        
        # Extract module and port definitions
        module_match = re.search(r'module\s+(\w+)\s*\((.*?)\);', rtl_code, re.DOTALL)
        if not module_match:
            return G
            
        module_name = module_match.group(1)
        ports = module_match.group(2)
        
        # Add module as central node
        G.add_node(module_name, type='module')
        
        # Add input and output ports
        input_ports = re.findall(r'input\s+.*?(\w+)\s*[,;]', rtl_code)
        output_ports = re.findall(r'output\s+.*?(\w+)\s*[,;]', rtl_code)
        
        for port in input_ports:
            G.add_node(port, type='input')
            G.add_edge(port, module_name)
            
        for port in output_ports:
            G.add_node(port, type='output')
            G.add_edge(module_name, port)
        
        # Find assign statements
        assigns = re.findall(r'assign\s+(\w+)\s*=\s*(.*?);', rtl_code)
        for target, expr in assigns:
            # Add assignment node
            assign_id = f"assign_{target}"
            G.add_node(assign_id, type='assign')
            
            # Add edge from assignment to target
            if target not in G:
                G.add_node(target, type='wire')
            G.add_edge(assign_id, target)
            
            # Add edges from sources to assignment
            sources = re.findall(r'\b(\w+)\b', expr)
            for source in sources:
                if source not in G and source != target:
                    G.add_node(source, type='wire')
                    G.add_edge(source, assign_id)
        
        # Find always blocks (simplified)
        always_blocks = re.findall(r'always\s*@\s*\((.*?)\)(.*?)(?=always|endmodule|$)', rtl_code, re.DOTALL)
        for i, (sensitivity, block_body) in enumerate(always_blocks):
            block_id = f"always_{i}"
            G.add_node(block_id, type='always')
            
            # Add sensitivity list connections
            sens_signals = re.findall(r'\b(\w+)\b', sensitivity)
            for signal in sens_signals:
                if signal not in G:
                    G.add_node(signal, type='wire')
                G.add_edge(signal, block_id)
            
            # Add outputs of the always block
            assignments = re.findall(r'(\w+)\s*<=', block_body)
            for target in assignments:
                if target not in G:
                    G.add_node(target, type='reg')
                G.add_edge(block_id, target)
                
            # Add inputs to the always block
            expressions = re.sub(r'\w+\s*<=.*?;', '', block_body)
            sources = re.findall(r'\b(\w+)\b', expressions)
            for source in sources:
                if source not in G and source not in assignments:
                    G.add_node(source, type='wire')
                    G.add_edge(source, block_id)
        
        return G

    def _estimate_combinational_depth(self, G):
        """
        Estimate combinational depth from the graph representation
        """
        # Find input and output nodes
        input_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'input']
        output_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'output']
        
        # Find maximum path length from inputs to outputs
        max_depth = 0
        for src in input_nodes:
            for dst in output_nodes:
                try:
                    path_length = len(nx.shortest_path(G, src, dst)) - 1
                    max_depth = max(max_depth, path_length)
                except nx.NetworkXNoPath:
                    continue
        
        return max_depth

    def _calculate_logical_effort(self, features):
        """
        Calculate logical effort based on circuit composition
        """
        # Approximate logical effort weights for different gate types
        logic_efforts = {
            'and_gate': 4/3,
            'or_gate': 5/3,
            'not_gate': 1,
            'xor_gate': 4,
            'buffer': 1,
            'mux': 2
        }
        
        # Estimate average logical effort based on logic gate percentages
        logical_effort = (
            features['logic_and_percent'] / 100 * logic_efforts['and_gate'] +
            features['logic_or_percent'] / 100 * logic_efforts['or_gate'] +
            features['logic_not_percent'] / 100 * logic_efforts['not_gate'] +
            features['logic_xor_percent'] / 100 * logic_efforts['xor_gate'] +
            features['logic_buffer_percent'] / 100 * logic_efforts['buffer']
        )
        
        return logical_effort

    def _preprocess_input(self, features):
        """
        Preprocess input features for model prediction
        """
        # Convert to DataFrame
        input_df = pd.DataFrame([features])
        
        # Feature engineering
        input_df['clock_period_ps'] = 1000000 / input_df['clock_frequency_mhz']
        input_df['path_to_required_ratio'] = input_df['data_path_delay_ps'] / input_df['required_time_ps']
        input_df['logic_to_sequential_ratio'] = input_df['combinational_logic'] / (input_df['logic_sequential_percent'] + 1)
        
        # Calculate logical effort
        logical_effort = self._calculate_logical_effort(features)
        input_df['logical_effort_estimate'] = logical_effort
        
        # Adjust delay per logic level based on logical effort
        input_df['depth_with_logical_effort'] = input_df['max_logic_depth'] * logical_effort
        input_df['delay_per_logic_level'] = input_df['data_path_delay_ps'] / np.maximum(input_df['max_logic_depth'], 1)
        
        # Create dummy variables for technology node
        input_df['technology'] = input_df['technology'].astype(str)
        input_df = pd.get_dummies(input_df, columns=['technology'], prefix='tech', drop_first=False)
        
        # Handle module name (if encoder is available)
        try:
            input_df['module_name_encoded'] = self.module_encoder.transform([features['module_name']])
        except:
            input_df['module_name_encoded'] = 0
            
        # Handle timing violation risk (if encoder is available)
        try:
            input_df['timing_violation_risk_encoded'] = self.risk_encoder.transform([features['timing_violation_risk']])
        except:
            input_df['timing_violation_risk_encoded'] = 0
            
        return input_df, logical_effort

    def _generate_recommendations(self, features, prediction_results, patterns):
        """
        Generate design recommendations based on prediction results and patterns
        """
        recommendations = []
        
        # Check timing violations
        if prediction_results['timing_violation_predicted']:
            violation_prob = prediction_results['timing_violation_probability']
            if violation_prob > 0.8:
                recommendations.append(f"{Fore.RED}CRITICAL: High probability of timing violations (>80%). Consider complete redesign.{Style.RESET_ALL}")
            
            # Check if it's due to large combinational depth
            if prediction_results['combinational_depth_predicted'] > 5:
                recommendations.append(f"{Fore.YELLOW}Pipeline your design: Break up long combinational paths by adding registers.{Style.RESET_ALL}")
                
            # Check for high delay per logic level
            if prediction_results['estimated_delay_per_level_ps'] > 200:
                recommendations.append(f"{Fore.YELLOW}Consider using faster gates or a more advanced technology node.{Style.RESET_ALL}")
                
            # Check logical effort
            if prediction_results['logical_effort_estimate'] > 2:
                recommendations.append(f"{Fore.YELLOW}Optimize gate types: Your design uses gates with high logical effort. Consider simplifying complex gates.{Style.RESET_ALL}")
                
            # Logic vs. sequential balance
            if features['logic_sequential_percent'] < 20:
                recommendations.append(f"{Fore.YELLOW}Add more pipelining: Your design is heavily combinational with minimal sequential logic.{Style.RESET_ALL}")
        else:
            recommendations.append(f"{Fore.GREEN}Design has met timing requirements with slack of approximately {features['slack_ps']:.2f}ps.{Style.RESET_ALL}")
            
            # Suggest optimizations
            if features['slack_ps'] > 500:
                recommendations.append(f"{Fore.CYAN}Power optimization potential: You have significant timing slack. Consider downsizing gates or reducing voltage to save power.{Style.RESET_ALL}")
                
        # Pattern-based recommendations
        if 'multiplier' in patterns and patterns['multiplier'] > 0:
            recommendations.append(f"{Fore.CYAN}Detected {patterns['multiplier']} multiplier patterns. For high-performance or low-power designs, consider using dedicated DSP blocks.{Style.RESET_ALL}")
            
        if 'pipeline' in patterns and patterns['pipeline'] > 3:
            recommendations.append(f"{Fore.CYAN}Your design uses a deeply pipelined architecture ({patterns['pipeline']}+ stages). Ensure the pipeline is balanced.{Style.RESET_ALL}")
            
        # Area recommendations
        if features['cell_count'] > 1000:
            recommendations.append(f"{Fore.YELLOW}Large design detected. Consider hierarchy-based optimization for better placement and routing.{Style.RESET_ALL}")
            
        # Additional recommendations for common patterns
        for pattern, count in patterns.items():
            if count > 5 and pattern in ['adder', 'mux']:
                recommendations.append(f"{Fore.CYAN}Multiple {pattern} instances detected ({count}). Consider using generator-based design to minimize redundancy.{Style.RESET_ALL}")
                
        # Technology-specific recommendations
        tech_node = int(features['technology']) if features['technology'].isdigit() else 45
        if tech_node < 22 and features['slack_ps'] < 100:
            recommendations.append(f"{Fore.YELLOW}At {tech_node}nm technology, consider leakage power and process variation impact on timing.{Style.RESET_ALL}")
            
        # Clock frequency recommendations
        if features['clock_frequency_mhz'] > 800 and features['slack_ps'] < 200:
            recommendations.append(f"{Fore.YELLOW}High clock frequency design ({features['clock_frequency_mhz']}MHz) with limited slack. Consider clock uncertainty and jitter analysis.{Style.RESET_ALL}")
            
        return recommendations

    def _plot_timing_analysis(self, results, output_file="timing_analysis.png"):
        """Generate visualization of timing analysis results"""
        try:
            # Create a new figure
            plt.figure(figsize=(12, 8))
            
            # Create a timing path diagram
            plt.subplot(2, 2, 1)
            
            # Get timing data
            delay = results['features']['data_path_delay_ps']
            required = results['features']['required_time_ps']
            slack = results['features']['slack_ps']
            
            # Plot timing bars
            bars = plt.barh(['Path Delay', 'Required Time'], [delay, required], color=['red', 'green'])
            
            # Add slack indicator
            if slack >= 0:
                plt.axvline(x=delay, color='blue', linestyle='--', label='Slack')
                plt.text(delay + slack/2, 0, f'Slack: {slack:.1f}ps', 
                         horizontalalignment='center', verticalalignment='center')
            else:
                plt.axvline(x=required, color='red', linestyle='--', label='Violation')
                plt.text(required + slack/2, 0, f'Violation: {-slack:.1f}ps', 
                         horizontalalignment='center', verticalalignment='center', color='red')
            
            plt.title('Timing Path Analysis')
            plt.xlabel('Time (ps)')
            
            # Plot gate distribution
            plt.subplot(2, 2, 2)
            gate_types = ['AND', 'OR', 'NOT', 'XOR', 'Buffer']
            gate_percentages = [
                results['features']['logic_and_percent'],
                results['features']['logic_or_percent'],
                results['features']['logic_not_percent'],
                results['features']['logic_xor_percent'],
                results['features']['logic_buffer_percent']
            ]
            
            plt.pie(gate_percentages, labels=gate_types, autopct='%1.1f%%')
            plt.title('Logic Gate Distribution')
            
            # Plot pattern distribution
            plt.subplot(2, 2, 3)
            if results['identified_patterns']:
                patterns = list(results['identified_patterns'].keys())
                pattern_counts = list(results['identified_patterns'].values())
                plt.bar(patterns, pattern_counts)
                plt.title('Identified Design Patterns')
                plt.xticks(rotation=45)
            else:
                plt.text(0.5, 0.5, 'No patterns identified', 
                         horizontalalignment='center', verticalalignment='center')
                plt.title('Pattern Analysis')
            
            # Add a heatmap showing timing risk by logic depth
            plt.subplot(2, 2, 4)
            
            # Create depth vs frequency data
            max_depth = results['combinational_depth_predicted']
            data = np.zeros((5, 5))
            
            # Fill with dummy data based on current design's depth and frequency
            for i in range(5):
                for j in range(5):
                    depth_factor = (i + 1) / 3  # Normalized around our current depth
                    freq_factor = (j + 1) / 3   # Normalized around our current frequency
                    
                    # Estimate slack based on depth and frequency
                    estimated_delay = results['features']['data_path_delay_ps'] * depth_factor
                    estimated_required = results['features']['required_time_ps'] / freq_factor
                    estimated_slack = estimated_required - estimated_delay
                    
                    # Convert to a heat value (negative=bad, positive=good)
                    if estimated_slack < 0:
                        data[i, j] = -min(1, abs(estimated_slack) / 500)  # Cap at -1
                    else:
                        data[i, j] = min(1, estimated_slack / 500)  # Cap at 1
            
            # Create heatmap
            im = plt.imshow(data, cmap='RdYlGn')
            plt.colorbar(im, label='Timing Slack')
            
            # Add labels
            plt.xticks(range(5), [f"{(i+1)*results['features']['clock_frequency_mhz']/3:.0f}MHz" for i in range(5)])
            plt.yticks(range(5), [f"{(i+1)*max_depth/3:.1f}x" for i in range(5)])
            
            plt.xlabel('Clock Frequency')
            plt.ylabel('Logic Depth Factor')
            plt.title('Timing Slack Sensitivity')
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure if output file is provided
            if output_file:
                plt.savefig(output_file)
                print(f"{Fore.GREEN}Timing analysis visualization saved to {output_file}{Style.RESET_ALL}")
                
            plt.close()
            return True
        except Exception as e:
            print(f"{Fore.RED}Error generating visualization: {str(e)}{Style.RESET_ALL}")
            return False

    def analyze_rtl(self, rtl_code, user_params=None):
        """
        Analyze RTL code and provide predictions and recommendations
        
        Args:
            rtl_code (str): The RTL code to analyze
            user_params (dict): User-specified parameters including clock_frequency, technology, etc.
        """
        print(f"\n{Fore.CYAN}Beginning RTL analysis...{Style.RESET_ALL}")
        
        # Progress bar for analysis steps
        steps = ['Extracting features', 'Identifying patterns', 'Building circuit graph', 
                 'Analyzing timing paths', 'Generating recommendations']
        
        with tqdm(total=len(steps), desc="Analysis Progress", bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Style.RESET_ALL)) as pbar:
            # Extract features from RTL with user parameters
            features = self._extract_rtl_features(rtl_code, user_params)
            # Extract features from RTL with user parameters
            features = self._extract_rtl_features(rtl_code, user_params)
            pbar.update(1)
            
            # Identify design patterns
            identified_patterns = self._identify_patterns(rtl_code)
            pbar.update(1)
            
            # Create graph representation
            circuit_graph = self._create_graph_representation(rtl_code)
            pbar.update(1)
            
            # Preprocess features for prediction
            input_df, logical_effort = self._preprocess_input(features)
            
            # Predict timing violation probability
            timing_violation_prob = 0.0
            timing_violation = False
            try:
                timing_violation_prob = self.timing_violation_model.predict_proba(input_df)[:, 1][0]
                timing_violation = timing_violation_prob > 0.5
            except:
                # Fallback if model prediction fails
                timing_violation = features['slack_ps'] < 0
                timing_violation_prob = 0.9 if timing_violation else 0.1
            
            # Predict combinational depth
            try:
                combinational_depth = self.combinational_depth_model.predict(input_df)[0]
            except:
                # Fallback if model prediction fails
                combinational_depth = self._estimate_combinational_depth(circuit_graph)
                if combinational_depth == 0:
                    combinational_depth = features['max_logic_depth']
            
            # Analyze timing paths
            delay_per_level = features['data_path_delay_ps'] / max(features['max_logic_depth'], 1)
            pbar.update(1)
            
            # Generate recommendations
            prediction_results = {
                'timing_violation_predicted': timing_violation,
                'timing_violation_probability': timing_violation_prob,
                'combinational_depth_predicted': combinational_depth,
                'estimated_delay_per_level_ps': delay_per_level,
                'logical_effort_estimate': logical_effort
            }
            
            recommendations = self._generate_recommendations(features, prediction_results, identified_patterns)
            pbar.update(1)
        
        # Prepare results
        results = {
            'features': features,
            'prediction_results': prediction_results,
            'recommendations': recommendations,
            'identified_patterns': identified_patterns,
            'combinational_depth_predicted': combinational_depth,
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Display results
        self._display_results(results)
        
        # Generate visualization
        self._plot_timing_analysis(results)
        
        return results
        
    def _display_results(self, results):
        """Display analysis results in a formatted way"""
        print(f"\n{Fore.CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Style.RESET_ALL}")
        print(f"{Fore.CYAN}╔══ TIMING ANALYSIS RESULTS ═══════════════════════════════════════╗{Style.RESET_ALL}")
        print(f"{Fore.CYAN}║{Style.RESET_ALL}")
        
        # Module info
        module_name = results['features']['module_name']
        print(f"{Fore.CYAN}║{Style.RESET_ALL} {Fore.YELLOW}Module:{Style.RESET_ALL} {module_name}")
        print(f"{Fore.CYAN}║{Style.RESET_ALL} {Fore.YELLOW}Technology:{Style.RESET_ALL} {results['features']['technology']}nm")
        print(f"{Fore.CYAN}║{Style.RESET_ALL} {Fore.YELLOW}Clock Frequency:{Style.RESET_ALL} {results['features']['clock_frequency_mhz']}MHz")
        print(f"{Fore.CYAN}║{Style.RESET_ALL}")
        
        # Timing summary
        print(f"{Fore.CYAN}║{Style.RESET_ALL} {Fore.YELLOW}TIMING SUMMARY{Style.RESET_ALL}")
        print(f"{Fore.CYAN}║{Style.RESET_ALL} Required Time:      {results['features']['required_time_ps']:.2f}ps")
        print(f"{Fore.CYAN}║{Style.RESET_ALL} Data Path Delay:    {results['features']['data_path_delay_ps']:.2f}ps")
        print(f"{Fore.CYAN}║{Style.RESET_ALL} Slack:              ", end="")
        
        slack = results['features']['slack_ps']
        if slack >= 0:
            print(f"{Fore.GREEN}{slack:.2f}ps (MET){Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}{slack:.2f}ps (VIOLATED){Style.RESET_ALL}")
        
        # Prediction results
        print(f"{Fore.CYAN}║{Style.RESET_ALL}")
        print(f"{Fore.CYAN}║{Style.RESET_ALL} {Fore.YELLOW}PREDICTION RESULTS{Style.RESET_ALL}")
        timing_violation_prob = results['prediction_results']['timing_violation_probability'] * 100
        
        print(f"{Fore.CYAN}║{Style.RESET_ALL} Timing Violation Probability: ", end="")
        if timing_violation_prob < 25:
            print(f"{Fore.GREEN}{timing_violation_prob:.1f}%{Style.RESET_ALL}")
        elif timing_violation_prob < 75:
            print(f"{Fore.YELLOW}{timing_violation_prob:.1f}%{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}{timing_violation_prob:.1f}%{Style.RESET_ALL}")
            
        print(f"{Fore.CYAN}║{Style.RESET_ALL} Predicted Combinational Depth: {results['combinational_depth_predicted']:.1f}")
        print(f"{Fore.CYAN}║{Style.RESET_ALL} Estimated Delay Per Level: {results['prediction_results']['estimated_delay_per_level_ps']:.2f}ps")
        print(f"{Fore.CYAN}║{Style.RESET_ALL}")
        
        # Design pattern detection
        print(f"{Fore.CYAN}║{Style.RESET_ALL} {Fore.YELLOW}DETECTED DESIGN PATTERNS{Style.RESET_ALL}")
        if results['identified_patterns']:
            for pattern, count in results['identified_patterns'].items():
                print(f"{Fore.CYAN}║{Style.RESET_ALL} • {pattern.replace('_', ' ').title()}: {count}")
        else:
            print(f"{Fore.CYAN}║{Style.RESET_ALL} • No specific patterns detected")
        
        print(f"{Fore.CYAN}║{Style.RESET_ALL}")
        
        # Circuit statistics
        print(f"{Fore.CYAN}║{Style.RESET_ALL} {Fore.YELLOW}CIRCUIT STATISTICS{Style.RESET_ALL}")
        print(f"{Fore.CYAN}║{Style.RESET_ALL} • Cell Count: {results['features']['cell_count']}")
        print(f"{Fore.CYAN}║{Style.RESET_ALL} • Combinational Logic: {results['features']['combinational_logic']}")
        print(f"{Fore.CYAN}║{Style.RESET_ALL} • Sequential Elements: {results['features']['rtl_sequential_block_count']}")
        print(f"{Fore.CYAN}║{Style.RESET_ALL} • Maximum Logic Depth: {results['features']['max_logic_depth']}")
        
        # Recommendations
        print(f"{Fore.CYAN}║{Style.RESET_ALL}")
        print(f"{Fore.CYAN}║{Style.RESET_ALL} {Fore.YELLOW}RECOMMENDATIONS{Style.RESET_ALL}")
        for i, rec in enumerate(results['recommendations'], 1):
            print(f"{Fore.CYAN}║{Style.RESET_ALL} {i}. {rec}")
        
        print(f"{Fore.CYAN}║{Style.RESET_ALL}")
        print(f"{Fore.CYAN}╚════════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}")
        print(f"{Fore.CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Style.RESET_ALL}\n")

    def export_report(self, results, format='json', filename=None):
        """
        Export analysis results to a file
        
        Args:
            results (dict): Analysis results to export
            format (str): Output format ('json', 'html', 'txt')
            filename (str): Output filename, defaults to 'rtl_timing_report.[format]'
        """
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rtl_timing_report_{timestamp}.{format}"
            
        print(f"{Fore.CYAN}Exporting report to {filename}...{Style.RESET_ALL}")
        
        try:
            if format == 'json':
                # Convert results to serializable format
                serializable_results = results.copy()
                # Handle numpy types
                for key, value in serializable_results['features'].items():
                    if isinstance(value, (np.int64, np.int32, np.float64, np.float32)):
                        serializable_results['features'][key] = float(value)
                
                with open(filename, 'w') as f:
                    json.dump(serializable_results, f, indent=2)
                    
            elif format == 'html':
                # Simple HTML report template
                html_content = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>RTL Timing Analysis Report</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        .container {{ max-width: 1000px; margin: 0 auto; }}
                        .header {{ background-color: #333; color: white; padding: 10px; text-align: center; }}
                        .section {{ margin: 20px 0; padding: 10px; border: 1px solid #ddd; }}
                        .met {{ color: green; font-weight: bold; }}
                        .violated {{ color: red; font-weight: bold; }}
                        .warning {{ color: orange; }}
                        table {{ border-collapse: collapse; width: 100%; }}
                        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                        th {{ background-color: #f2f2f2; }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <div class="header">
                            <h1>RTL Timing Analysis Report</h1>
                            <p>Generated on {results['timestamp']}</p>
                        </div>
                        
                        <div class="section">
                            <h2>Module Information</h2>
                            <table>
                                <tr><th>Module Name</th><td>{results['features']['module_name']}</td></tr>
                                <tr><th>Technology</th><td>{results['features']['technology']}nm</td></tr>
                                <tr><th>Clock Frequency</th><td>{results['features']['clock_frequency_mhz']}MHz</td></tr>
                            </table>
                        </div>
                        
                        <div class="section">
                            <h2>Timing Summary</h2>
                            <table>
                                <tr><th>Required Time</th><td>{results['features']['required_time_ps']:.2f}ps</td></tr>
                                <tr><th>Data Path Delay</th><td>{results['features']['data_path_delay_ps']:.2f}ps</td></tr>
                                <tr><th>Slack</th>
                                    <td class="{'met' if results['features']['slack_ps'] >= 0 else 'violated'}">
                                        {results['features']['slack_ps']:.2f}ps 
                                        ({('MET' if results['features']['slack_ps'] >= 0 else 'VIOLATED')})
                                    </td>
                                </tr>
                            </table>
                        </div>
                        
                        <div class="section">
                            <h2>Prediction Results</h2>
                            <table>
                                <tr>
                                    <th>Timing Violation Probability</th>
                                    <td class="{'met' if results['prediction_results']['timing_violation_probability'] < 0.25 else 'warning' if results['prediction_results']['timing_violation_probability'] < 0.75 else 'violated'}">
                                        {results['prediction_results']['timing_violation_probability'] * 100:.1f}%
                                    </td>
                                </tr>
                                <tr>
                                    <th>Predicted Combinational Depth</th>
                                    <td>{results['combinational_depth_predicted']:.1f}</td>
                                </tr>
                                <tr>
                                    <th>Estimated Delay Per Level</th>
                                    <td>{results['prediction_results']['estimated_delay_per_level_ps']:.2f}ps</td>
                                </tr>
                            </table>
                        </div>
                        
                        <div class="section">
                            <h2>Detected Design Patterns</h2>
                            <table>
                                <tr><th>Pattern</th><th>Count</th></tr>
                """
                
                # Add pattern rows
                if results['identified_patterns']:
                    for pattern, count in results['identified_patterns'].items():
                        html_content += f"<tr><td>{pattern.replace('_', ' ').title()}</td><td>{count}</td></tr>\n"
                else:
                    html_content += "<tr><td colspan='2'>No specific patterns detected</td></tr>\n"
                
                # Add circuit statistics and recommendations
                html_content += f"""
                            </table>
                        </div>
                        
                        <div class="section">
                            <h2>Circuit Statistics</h2>
                            <table>
                                <tr><th>Cell Count</th><td>{results['features']['cell_count']}</td></tr>
                                <tr><th>Combinational Logic</th><td>{results['features']['combinational_logic']}</td></tr>
                                <tr><th>Sequential Elements</th><td>{results['features']['rtl_sequential_block_count']}</td></tr>
                                <tr><th>Maximum Logic Depth</th><td>{results['features']['max_logic_depth']}</td></tr>
                            </table>
                        </div>
                        
                        <div class="section">
                            <h2>Recommendations</h2>
                            <ol>
                """
                
                # Add recommendations (strip ANSI color codes)
                for rec in results['recommendations']:
                    # Remove ANSI color codes
                    clean_rec = re.sub(r'\x1B\[[0-9;]*[a-zA-Z]', '', rec)
                    html_content += f"<li>{clean_rec}</li>\n"
                
                html_content += """
                            </ol>
                        </div>
                    </div>
                </body>
                </html>
                """
                
                with open(filename, 'w') as f:
                    f.write(html_content)
                    
            elif format == 'txt':
                with open(filename, 'w') as f:
                    # Basic text report
                    f.write("=== RTL TIMING ANALYSIS REPORT ===\n")
                    f.write(f"Generated on: {results['timestamp']}\n\n")
                    
                    # Module info
                    f.write("MODULE INFORMATION\n")
                    f.write(f"Module Name: {results['features']['module_name']}\n")
                    f.write(f"Technology: {results['features']['technology']}nm\n")
                    f.write(f"Clock Frequency: {results['features']['clock_frequency_mhz']}MHz\n\n")
                    
                    # Timing summary
                    f.write("TIMING SUMMARY\n")
                    f.write(f"Required Time: {results['features']['required_time_ps']:.2f}ps\n")
                    f.write(f"Data Path Delay: {results['features']['data_path_delay_ps']:.2f}ps\n")
                    slack = results['features']['slack_ps']
                    f.write(f"Slack: {slack:.2f}ps ({'MET' if slack >= 0 else 'VIOLATED'})\n\n")
                    
                    # Prediction results
                    f.write("PREDICTION RESULTS\n")
                    f.write(f"Timing Violation Probability: {results['prediction_results']['timing_violation_probability'] * 100:.1f}%\n")
                    f.write(f"Predicted Combinational Depth: {results['combinational_depth_predicted']:.1f}\n")
                    f.write(f"Estimated Delay Per Level: {results['prediction_results']['estimated_delay_per_level_ps']:.2f}ps\n\n")
                    
                    # Design patterns
                    f.write("DETECTED DESIGN PATTERNS\n")
                    if results['identified_patterns']:
                        for pattern, count in results['identified_patterns'].items():
                            f.write(f"- {pattern.replace('_', ' ').title()}: {count}\n")
                    else:
                        f.write("- No specific patterns detected\n")
                    f.write("\n")
                    
                    # Circuit statistics
                    f.write("CIRCUIT STATISTICS\n")
                    f.write(f"Cell Count: {results['features']['cell_count']}\n")
                    f.write(f"Combinational Logic: {results['features']['combinational_logic']}\n")
                    f.write(f"Sequential Elements: {results['features']['rtl_sequential_block_count']}\n")
                    f.write(f"Maximum Logic Depth: {results['features']['max_logic_depth']}\n\n")
                    
                    # Recommendations
                    f.write("RECOMMENDATIONS\n")
                    for i, rec in enumerate(results['recommendations'], 1):
                        # Remove ANSI color codes
                        clean_rec = re.sub(r'\x1B\[[0-9;]*[a-zA-Z]', '', rec)
                        f.write(f"{i}. {clean_rec}\n")
            
            print(f"{Fore.GREEN}Report exported successfully to {filename}{Style.RESET_ALL}")
            return True
            
        except Exception as e:
            print(f"{Fore.RED}Error exporting report: {str(e)}{Style.RESET_ALL}")
            return False

def main():
    """Main function for CLI usage"""
    parser = argparse.ArgumentParser(description='RTL Timing Analyzer')
    parser.add_argument('--file', '-f', type=str, help='RTL file to analyze')
    parser.add_argument('--clock', '-c', type=float, default=500, help='Clock frequency in MHz')
    parser.add_argument('--tech', '-t', type=str, default='45', help='Technology node in nm')
    parser.add_argument('--report', '-r', type=str, choices=['json', 'html', 'txt'], default='html', 
                        help='Report format to generate')
    parser.add_argument('--output', '-o', type=str, help='Output file name')
    parser.add_argument('--plot', '-p', action='store_true', help='Generate timing analysis plot')
    
    args = parser.parse_args()
    
    if args.file:
        try:
            # Initialize analyzer
            analyzer = RTLTimingAnalyzer()
            
            # Read RTL file
            with open(args.file, 'r') as f:
                rtl_code = f.read()
                
            # Set up user parameters
            user_params = {
                'clock_frequency_mhz': args.clock,
                'technology': args.tech
            }
            
            # Run analysis
            print(f"{Fore.CYAN}Analyzing RTL file: {args.file}{Style.RESET_ALL}")
            results = analyzer.analyze_rtl(rtl_code, user_params)
            
            # Export report if requested
            if args.report:
                analyzer.export_report(results, format=args.report, filename=args.output)
                
        except Exception as e:
            print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
            return 1
    else:
        # Interactive mode
        print(f"{Fore.YELLOW}No RTL file specified. Running in interactive mode.{Style.RESET_ALL}")
        analyzer = RTLTimingAnalyzer()
        
        # Ask for RTL input
        print(f"{Fore.CYAN}Enter or paste your RTL code (type 'END' on a new line when finished):{Style.RESET_ALL}")
        rtl_lines = []
        while True:
            line = input()
            if line.strip() == 'END':
                break
            rtl_lines.append(line)
        
        rtl_code = '\n'.join(rtl_lines)
        
        if not rtl_code.strip():
            print(f"{Fore.RED}No RTL code provided. Exiting.{Style.RESET_ALL}")
            return 1
        
        # Ask for parameters
        try:
            clock_freq = float(input(f"{Fore.CYAN}Enter clock frequency in MHz [default: 500]: {Style.RESET_ALL}") or "500")
            tech_node = input(f"{Fore.CYAN}Enter technology node in nm [default: 45]: {Style.RESET_ALL}") or "45"
            
            user_params = {
                'clock_frequency_mhz': clock_freq,
                'technology': tech_node
            }
            
            # Run analysis
            results = analyzer.analyze_rtl(rtl_code, user_params)
            
            # Ask if user wants to export a report
            export_report = input(f"{Fore.CYAN}Export report? (y/n) [default: y]: {Style.RESET_ALL}").lower() != 'n'
            if export_report:
                report_format = input(f"{Fore.CYAN}Report format (json, html, txt) [default: html]: {Style.RESET_ALL}") or "html"
                filename = input(f"{Fore.CYAN}Output filename [default: auto-generated]: {Style.RESET_ALL}")
                analyzer.export_report(results, format=report_format, filename=filename or None)
                
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Analysis canceled.{Style.RESET_ALL}")
            return 1
        except Exception as e:
            print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
            return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
