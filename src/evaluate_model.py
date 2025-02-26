import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, Tuple

from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    r2_score, 
    max_error, 
    mean_absolute_percentage_error,
    explained_variance_score
)
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('model_evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_model_and_data(
    model_path: str, 
    test_data_path: str
) -> Tuple[object, pd.DataFrame, pd.Series]:
    """
    Load trained model and test data.
    
    Parameters:
    -----------
    model_path : str
        Path to the trained model
    test_data_path : str
        Path to the test data CSV
    
    Returns:
    --------
    Tuple of (model, X_test, y_test)
    """
    try:
        # Validate file existence
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        if not os.path.exists(test_data_path):
            raise FileNotFoundError(f"Test data not found at {test_data_path}")
        
        # Load model
        model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
        
        # Load test data
        test_df = pd.read_csv(test_data_path)
        logger.info(f"Test data loaded. Shape: {test_df.shape}")
        
        # Prepare features and target
        X_test = test_df.drop(['combinational_depth', 'module_name', 'signal_name'], axis=1, errors='ignore')
        y_test = test_df['combinational_depth']
        
        return model, X_test, y_test
    
    except Exception as e:
        logger.error(f"Error loading model or data: {e}")
        raise

def calculate_metrics(y_test: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive model performance metrics.
    
    Parameters:
    -----------
    y_test : pd.Series
        True target values
    y_pred : np.ndarray
        Predicted target values
    
    Returns:
    --------
    Dict of performance metrics
    """
    metrics = {
        'Mean Absolute Error (MAE)': mean_absolute_error(y_test, y_pred),
        'Root Mean Squared Error (RMSE)': np.sqrt(mean_squared_error(y_test, y_pred)),
        'Mean Absolute Percentage Error (MAPE)': mean_absolute_percentage_error(y_test, y_pred),
        'R² Score': r2_score(y_test, y_pred),
        'Max Error': max_error(y_test, y_pred),
        'Explained Variance Score': explained_variance_score(y_test, y_pred)
    }
    
    return metrics

def plot_error_distribution(errors: np.ndarray, output_path: str = 'error_distribution.png'):
    """
    Plot the distribution of prediction errors.
    
    Parameters:
    -----------
    errors : np.ndarray
        Prediction errors
    output_path : str
        Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, kde=True)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title('Prediction Error Distribution')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_actual_vs_predicted(
    y_test: pd.Series, 
    y_pred: np.ndarray, 
    output_path: str = 'actual_vs_predicted.png'
):
    """
    Create a scatter plot of actual vs predicted values.
    
    Parameters:
    -----------
    y_test : pd.Series
        True target values
    y_pred : np.ndarray
        Predicted target values
    output_path : str
        Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect Prediction')
    plt.xlabel('Actual Combinational Depth')
    plt.ylabel('Predicted Combinational Depth')
    plt.title('Actual vs Predicted Combinational Depth')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def analyze_performance_by_category(
    test_df: pd.DataFrame, 
    y_test: pd.Series, 
    y_pred: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """
    Analyze model performance across different categories.
    
    Parameters:
    -----------
    test_df : pd.DataFrame
        Full test dataframe
    y_test : pd.Series
        True target values
    y_pred : np.ndarray
        Predicted target values
    
    Returns:
    --------
    Dict of performance metrics by category
    """
    performance_by_category = {}
    
    # Analyze by module type
    if 'module_type' in test_df.columns:
        module_results = {}
        for module_type in test_df['module_type'].unique():
            module_idx = test_df['module_type'] == module_type
            module_mae = mean_absolute_error(y_test[module_idx], y_pred[module_idx])
            module_results[module_type] = module_mae
        performance_by_category['Module Type'] = module_results
    
    # Analyze by technology node
    if 'technology_node_nm' in test_df.columns:
        tech_results = {}
        for tech_node in sorted(test_df['technology_node_nm'].unique()):
            tech_idx = test_df['technology_node_nm'] == tech_node
            tech_mae = mean_absolute_error(y_test[tech_idx], y_pred[tech_idx])
            tech_results[f"{tech_node}nm"] = tech_mae
        performance_by_category['Technology Node'] = tech_results
    
    return performance_by_category

def generate_detailed_report(
    metrics: Dict[str, float], 
    category_performance: Dict[str, Dict[str, float]]
) -> None:
    """
    Generate a comprehensive JSON report of model performance.
    
    Parameters:
    -----------
    metrics : Dict[str, float]
        Global performance metrics
    category_performance : Dict[str, Dict[str, float]]
        Performance across different categories
    """
    report = {
        "Global Metrics": metrics,
        "Category Performance": category_performance
    }
    
    with open('model_performance_report.json', 'w') as f:
        json.dump(report, f, indent=4)
    
    logger.info("Detailed performance report generated.")

def evaluate_model(
    model_path: str = 'xgb_depth_predictor.joblib', 
    test_data_path: str = 'test_dataset2.csv'
) -> Optional[Dict[str, float]]:
    """
    Comprehensive model evaluation function.
    
    Parameters:
    -----------
    model_path : str
        Path to the trained model
    test_data_path : str
        Path to the test dataset
    
    Returns:
    --------
    Dict of performance metrics or None if evaluation fails
    """
    try:
        # Load model and data
        model, X_test, y_test = load_model_and_data(model_path, test_data_path)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = calculate_metrics(y_test, y_pred)
        
        # Log metrics
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        # Visualizations
        plot_error_distribution(y_pred - y_test)
        plot_actual_vs_predicted(y_test, y_pred)
        
        # Analyze performance by category
        category_performance = analyze_performance_by_category(
            pd.read_csv(test_data_path), y_test, y_pred
        )
        
        # Generate detailed report
        generate_detailed_report(metrics, category_performance)
        
        return metrics
    
    except Exception as e:
        logger.error(f"Model evaluation failed: {e}")
        return None

def main():
    """
    Main execution function for model evaluation.
    """
    try:
        # Evaluate the model
        metrics = evaluate_model()
        
        if metrics:
            print("Model evaluation completed successfully.")
        else:
            print("Model evaluation encountered an error.")
    
    except Exception as e:
        logger.error(f"Unexpected error during evaluation: {e}")

if __name__ == "__main__":
    main()
