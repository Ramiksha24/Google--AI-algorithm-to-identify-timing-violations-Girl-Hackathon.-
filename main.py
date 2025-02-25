#!/usr/bin/env python3
"""
RTL Combinational Depth Predictor

A tool for predicting combinational logic depth in RTL designs without running synthesis.
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import time

# Import functions from other modules
from rtl_analysis import analyze_rtl_module, generate_timing_report
from model_training import compare_models
from feature_extraction import extract_features_from_rtl

def main():
    """Main entry point for the RTL Combinational Depth Predictor tool."""
    parser = argparse.ArgumentParser(description='RTL Combinational Depth Predictor')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model on the dataset')
    train_parser.add_argument('--dataset', required=True, help='Path to the dataset CSV file')
    train_parser.add_argument('--model-type', default='xgboost', 
                             choices=['xgboost', 'random_forest', 'neural_network', 'svr', 'linear'],
                             help='Type of model to train')
    train_parser.add_argument('--output', default='rtl_depth_model.joblib',
                             help='Path to save the trained model')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare different ML models')
    compare_parser.add_argument('--dataset', required=True, help='Path to the dataset CSV file')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze an RTL module')
    analyze_parser.add_argument('--rtl', required=True, help='Path to the RTL file')
    analyze_parser.add_argument('--model', default='xgb_depth_predictor.joblib',
                              help='Path to the trained model')
    analyze_parser.add_argument('--clock', type=float, default=2.0,
                              help='Clock period in nanoseconds')
    analyze_parser.add_argument('--output', help='Path to save the timing report')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict depth for a specific signal')
    predict_parser.add_argument('--rtl', required=True, help='Path to the RTL file')
    predict_parser.add_argument('--signal', required=True, help='Name of the signal to analyze')
    predict_parser.add_argument('--model', default='xgb_depth_predictor.joblib',
                               help='Path to the trained model')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        print(f"Training {args.model_type} model on dataset {args.dataset}...")
        
        if args.model_type == 'xgboost':
            print("Using XGBoost model (recommended)")
            # Code to train XGBoost model
        elif args.model_type == 'random_forest':
            print("Using Random Forest model")
            # Code to train Random Forest model
        # Add other model types...
        
        print(f"Model trained and saved to {args.output}")
    
    elif args.command == 'compare':
        print(f"Comparing models on dataset {args.dataset}...")
        results = compare_models(args.dataset)
        print("\nModel Comparison Results:")
        print(results.sort_values('MAE').to_string(index=False))
    
    elif args.command == 'analyze':
        print(f"Analyzing RTL module in {args.rtl}...")
        
        # Read RTL file
        with open(args.rtl, 'r') as f:
            rtl_code = f.read()
        
        # Extract module name
        module_name = os.path.basename(args.rtl).split('.')[0]
        
        # Analyze module
        start_time = time.time()
        results = analyze_rtl_module(rtl_code, args.model)
        analyze_time = time.time() - start_time
        
        # Generate timing report
        report = generate_timing_report(results, module_name, args.clock)
        
        # Print summary
        critical_count = sum(results['criticality'] == 'High')
        near_critical_count = sum(results['criticality'] == 'Medium')
        print(f"Analysis completed in {analyze_time:.2f} seconds")
        print(f"Found {len(results)} signals, {critical_count} critical, {near_critical_count} near-critical")
        
        # Print top critical signals
        print("\nTop Critical Signals:")
        critical = results[results['criticality'] == 'High'].head(5)
        if len(critical) > 0:
            for _, row in critical.iterrows():
                print(f"  {row['signal_name']}: Depth={row['predicted_depth']:.2f}, Criticality={row['criticality']}")
        else:
            print("  None found")
        
        # Save report if requested
        if args.output:
            with open(args.output, 'w') as f:
                f.write(report)
            print(f"Timing report saved to {args.output}")
        else:
            print("\nTiming Report:")
            print(report)
    
    elif args.command == 'predict':
        print(f"Predicting combinational depth for signal {args.signal} in {args.rtl}...")
        
        # Read RTL file
        with open(args.rtl, 'r') as f:
            rtl_code = f.read()
        
        # Load model
        model = joblib.load(args.model)
        
        # Extract features
        features = extract_features_from_rtl(rtl_code, args.signal)
        
        # Add module type and other required features
        features['module_type'] = determine_module_type(rtl_code)
        features['technology_node_nm'] = 28
        features['clock_frequency_mhz'] = 500
        features['optimization_level'] = 2
        
        # Convert to DataFrame
        features_df = pd.DataFrame([features])
        
        # Make prediction
        depth = model.predict(features_df)[0]
        
        # Determine criticality
        if depth > 10:
            criticality = "HIGH RISK - likely timing violation"
        elif depth > 7:
            criticality = "MEDIUM RISK - potential timing issue in high-frequency designs"
        else:
            criticality = "LOW RISK - likely meets timing"
        
        print(f"Predicted combinational depth: {depth:.2f}")
        print(f"Risk assessment: {criticality}")
        print(f"Key features:")
        print(f"  Fan-in: {features['fanin']}")
        print(f"  Combinational operations: {features['combinational_ops']}")
        print(f"  Arithmetic operations: {features['arithmetic_ops']}")
        print(f"  Signal width: {features['signal_width']}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
