import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import xgboost as xgb
import joblib
import networkx as nx
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    # Remove empty rows at the end
    df = df.dropna(how='all')
    print(f"Dataset shape: {df.shape}")
    return df

# Data preprocessing with enhanced features
def preprocess_data(df):
    # Check for missing values
    print("Missing values:\n", df.isnull().sum())
    
    # Convert timing_violation to binary (1/0)
    if 'timing_violation' in df.columns:
        df['timing_violation'] = df['timing_violation'].map({'YES': 1, 'NO': 0, 'Yes': 1, 'No': 0})
    
    # Convert timing_violation_risk to categorical
    if 'timing_violation_risk' in df.columns:
        # Create a risk_encoder for deployment
        risk_encoder = LabelEncoder()
        df['timing_violation_risk_encoded'] = risk_encoder.fit_transform(df['timing_violation_risk'])
        joblib.dump(risk_encoder, 'risk_encoder.pkl')
    
    # Encode module_name
    label_encoder = LabelEncoder()
    df['module_name_encoded'] = label_encoder.fit_transform(df['module_name'])
    
    # Save label encoder for deployment
    joblib.dump(label_encoder, 'module_name_encoder.pkl')
    
    # Feature engineering
    df['clock_period_ps'] = 1000000 / df['clock_frequency_mhz']  # Convert MHz to ps period
    df['path_to_required_ratio'] = df['data_path_delay_ps'] / df['required_time_ps']
    
    # Handle potential division by zero
    df['logic_to_sequential_ratio'] = df['combinational_logic'] / (df['logic_sequential_percent'] + 1)  # Adding 1 to avoid division by zero
    
    # NEW: Add logical effort calculations based on circuit composition
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
    df['logical_effort_estimate'] = (
        df['logic_and_percent'] / 100 * logic_efforts['and_gate'] +
        df['logic_or_percent'] / 100 * logic_efforts['or_gate'] +
        df['logic_not_percent'] / 100 * logic_efforts['not_gate'] +
        df['logic_xor_percent'] / 100 * logic_efforts['xor_gate'] +
        df['logic_buffer_percent'] / 100 * logic_efforts['buffer']
    )
    
    # Calculate adjusted combinational depth based on logical effort
    df['depth_with_logical_effort'] = df['combinational_depth'] * df['logical_effort_estimate']
    
    # Calculate path delay per logic level
    df['delay_per_logic_level'] = df['data_path_delay_ps'] / np.maximum(df['combinational_depth'], 1)
    
    # Convert technology to categorical
    df['technology'] = df['technology'].astype(str)
    df = pd.get_dummies(df, columns=['technology'], prefix='tech', drop_first=False)
    
    return df

# NEW: Pattern recognition for sub-block identification 
def identify_repeated_patterns(rtl_design):
    """
    Identify repeated structural patterns in RTL design
    Returns dictionary of patterns and their frequency
    
    This is a simplified representation of what would be a more complex
    graph-based pattern matching algorithm in a real implementation
    """
    # This would be implemented with a real RTL parser and NetworkX
    # Here we simulate the functionality
    
    # Create a graph representation of the RTL design
    # In a real implementation, this would parse RTL code
    G = nx.DiGraph()
    
    # Simplified pattern recognition
    # In reality, this would use subgraph isomorphism algorithms
    patterns = defaultdict(int)
    
    # Example patterns (simplified for demonstration)
    if 'multiplier' in rtl_design.lower():
        patterns['multiplier'] = rtl_design.count('*')
    if 'adder' in rtl_design.lower():
        patterns['adder'] = rtl_design.count('+')
    if 'alu' in rtl_design.lower():
        patterns['alu'] = 1
        
    return patterns

# Feature selection
def select_features(df, target_type):
    # Columns to drop for both models
    columns_to_drop = ['module_name', 'timing_violation_risk']
    
    if target_type == 'classification':
        # For timing violation prediction
        target = 'timing_violation'
        # Additional columns to drop for classification
        if 'timing_violation_count' in df.columns:
            columns_to_drop.append('timing_violation_count')
        if 'worst_violation' in df.columns:
            columns_to_drop.append('worst_violation')
    else:
        # For combinational depth prediction
        target = 'combinational_depth'
        if target not in df.columns:
            print(f"ERROR: Target column '{target}' not found in DataFrame.")
            print("Available columns:", df.columns.tolist())
            raise KeyError(f"Target column '{target}' not found")
    
    # Create features and target
    X = df.drop(columns=columns_to_drop + [target], errors='ignore')
    y = df[target]
    
    print(f"Features shape for {target}: {X.shape}")
    return X, y

# Enhanced exploratory data analysis
def perform_eda(df):
    # Distribution of target variables
    plt.figure(figsize=(12, 5))
    
    # Timing violation distribution
    plt.subplot(1, 2, 1)
    sns.countplot(x='timing_violation', data=df)
    plt.title('Distribution of Timing Violations')
    
    # Combinational depth distribution
    plt.subplot(1, 2, 2)
    sns.histplot(df['combinational_depth'], kde=True)
    plt.title('Distribution of Combinational Depth')
    
    plt.tight_layout()
    plt.savefig('target_distributions.png')
    
    # NEW: Plot the logical effort vs combinational depth
    plt.figure(figsize=(10, 6))
    plt.scatter(df['combinational_depth'], df['depth_with_logical_effort'])
    plt.xlabel('Raw Combinational Depth')
    plt.ylabel('Depth Adjusted with Logical Effort')
    plt.title('Impact of Logical Effort on Path Depth')
    plt.savefig('logical_effort_impact.png')
    
    # Correlation matrix
    plt.figure(figsize=(14, 12))
    numeric_cols = df.select_dtypes(include=['number']).columns
    correlation = df[numeric_cols].corr()
    sns.heatmap(correlation, annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.savefig('correlation_matrix.png')

    plt.figure(figsize=(15, 10))
    numeric_cols = df.select_dtypes(include=['number']).columns[:10]  # First 10 numeric columns
    for i, col in enumerate(numeric_cols):
        plt.subplot(4, 3, i+1)
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.tight_layout()
    plt.savefig('feature_distributions.png')
    
    # Create a pairplot for key features
    sns.pairplot(df[numeric_cols])
    plt.savefig('feature_pairplot.png')
    
    return correlation

# Build and train XGBoost model for classification (timing violation)
def build_classification_model(X_train, y_train, X_test, y_test):
    # Define the model pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('xgb', xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            tree_method='hist',  # For scalability with large datasets
            n_jobs=-1  # Use all available cores
        ))
    ])
    
    # Hyperparameter tuning
    param_grid = {
        'xgb__n_estimators': [50, 100, 200],
        'xgb__max_depth': [3, 5, 7],
        'xgb__learning_rate': [0.01, 0.1, 0.2],
        'xgb__subsample': [0.8, 1.0],
        'xgb__colsample_bytree': [0.8, 1.0]
    }
    
    # Use stratified k-fold for imbalanced classification
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    # Train the model
    grid_search.fit(X_train, y_train)
    
    # Best model
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    
    # Save the model
    joblib.dump(best_model, 'timing_violation_model.pkl')
    joblib.dump(X_train.columns.tolist(), 'timing_feature_names.pkl')
    
    # Evaluate the model
    y_pred = best_model.predict(X_test)
    
    # Classification metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"Classification Performance Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Feature importance
    feature_importance = best_model.named_steps['xgb'].feature_importances_
    feature_names = X_train.columns
    
    # Sort feature importance
    indices = np.argsort(feature_importance)[::-1]
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    plt.title("Feature Importance for Timing Violation Prediction")
    plt.bar(range(X_train.shape[1]), feature_importance[indices])
    plt.xticks(range(X_train.shape[1]), feature_names[indices], rotation=90)
    plt.tight_layout()
    plt.savefig('feature_importance_classification.png')
    
    return best_model, feature_importance, feature_names

# Enhanced regression model for combinational depth prediction
def build_regression_model(X_train, y_train, X_test, y_test):
    # Define the model pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('xgb', xgb.XGBRegressor(
            objective='reg:squarederror',
            tree_method='hist',  # For scalability
            n_jobs=-1  # Use all available cores
        ))
    ])
    
    # Hyperparameter tuning
    param_grid = {
        'xgb__n_estimators': [50, 100, 200],
        'xgb__max_depth': [3, 5, 7],
        'xgb__learning_rate': [0.01, 0.1, 0.2],
        'xgb__subsample': [0.8, 1.0],
        'xgb__colsample_bytree': [0.8, 1.0]
    }
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    # Train the model
    grid_search.fit(X_train, y_train)
    
    # Best model
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    
    # Save the model
    joblib.dump(best_model, 'combinational_depth_model.pkl')
    joblib.dump(X_train.columns.tolist(), 'depth_feature_names.pkl')
    
    # Evaluate the model
    y_pred = best_model.predict(X_test)
    
    # Regression metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Regression Performance Metrics:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Feature importance
    feature_importance = best_model.named_steps['xgb'].feature_importances_
    feature_names = X_train.columns
    
    # Sort feature importance
    indices = np.argsort(feature_importance)[::-1]
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    plt.title("Feature Importance for Combinational Depth Prediction")
    plt.bar(range(X_train.shape[1]), feature_importance[indices])
    plt.xticks(range(X_train.shape[1]), feature_names[indices], rotation=90)
    plt.tight_layout()
    plt.savefig('feature_importance_regression.png')
    
    return best_model, feature_importance, feature_names

# NEW: Scalable prediction function that handles sub-blocks
def create_scalable_predictor():
    # Load the trained models
    timing_violation_model = joblib.load('timing_violation_model.pkl')
    combinational_depth_model = joblib.load('combinational_depth_model.pkl')
    module_encoder = joblib.load('module_name_encoder.pkl')
    risk_encoder = joblib.load('risk_encoder.pkl')

    try:
        timing_feature_names = joblib.load('timing_feature_names.pkl')
        depth_feature_names = joblib.load('depth_feature_names.pkl')
    except:
        print("Feature names files not found. Using default features.")
        # Define basic features that should be in both models
        timing_feature_names = []  # These will need to be populated with actual column names if the files aren't found
        depth_feature_names = []
    
    def predict_with_pattern_analysis(input_data, rtl_content=None):
        """
        Make scalable predictions for new circuit designs with pattern recognition
        
        Parameters:
        input_data (dict): Dictionary with circuit design parameters
        rtl_content (str): RTL code content for pattern analysis (optional)
        
        Returns:
        dict: Predictions for timing violation and combinational depth, with additional insights
        """
        # Start with base prediction
        prediction_result = predict_base(input_data)
        
        # If RTL content is provided, enhance prediction with pattern recognition
        if rtl_content:
            # Identify repeated patterns
            patterns = identify_repeated_patterns(rtl_content)
            
            # Count total patterns
            total_patterns = sum(patterns.values())
            
            # Add pattern information to results
            prediction_result['identified_patterns'] = patterns
            
            # If patterns found, adjust predictions for large designs
            if total_patterns > 0:
                # Estimate scalability impact
                # In a real implementation, this would use more sophisticated methods
                prediction_result['estimated_processing_speedup'] = f"{total_patterns}x (by analyzing unique patterns once)"
                
                # Add insights about design complexity
                prediction_result['design_complexity_analysis'] = {
                    'repeated_structures': patterns,
                    'unique_structure_count': len(patterns),
                    'total_repeated_instances': total_patterns
                }
        
        return prediction_result
    
    def predict_base(input_data):
        """Base prediction function for single module"""
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Preprocess the input
        # Encode module name
        input_df['module_name_encoded'] = module_encoder.transform([input_data['module_name']])
        
        # Encode risk if present
        if 'timing_violation_risk' in input_data:
            input_df['timing_violation_risk_encoded'] = risk_encoder.transform([input_data['timing_violation_risk']])
        
        # Feature engineering
        input_df['clock_period_ps'] = 1000000 / input_df['clock_frequency_mhz']
        input_df['path_to_required_ratio'] = input_df['data_path_delay_ps'] / input_df['required_time_ps']
        input_df['logic_to_sequential_ratio'] = input_df['combinational_logic'] / (input_df['logic_sequential_percent'] + 1)
        
        # Add logical effort calculation
        logic_efforts = {
            'and_gate': 4/3, 'or_gate': 5/3, 'not_gate': 1,
            'xor_gate': 4, 'buffer': 1, 'mux': 2
        }
        
        input_df['logical_effort_estimate'] = (
            input_df['logic_and_percent'] / 100 * logic_efforts['and_gate'] +
            input_df['logic_or_percent'] / 100 * logic_efforts['or_gate'] +
            input_df['logic_not_percent'] / 100 * logic_efforts['not_gate'] +
            input_df['logic_xor_percent'] / 100 * logic_efforts['xor_gate'] +
            input_df['logic_buffer_percent'] / 100 * logic_efforts['buffer']
        )
        
        # For delay_per_logic_level, use max_logic_depth temporarily
        input_df['delay_per_logic_level'] = input_df['data_path_delay_ps'] / np.maximum(input_df['max_logic_depth'], 1)
        
        # Create dummy variables for technology node
        input_df['technology'] = input_df['technology'].astype(str)
        input_df = pd.get_dummies(input_df, columns=['technology'], prefix='tech', drop_first=False)
        
        # Create separate DataFrames for each model with exactly the right columns
        input_df_timing = pd.DataFrame(index=input_df.index)
        input_df_depth = pd.DataFrame(index=input_df.index)
        
        # Fill with the correct features in the correct order
        for feature in timing_feature_names:
            if feature in input_df.columns:
                input_df_timing[feature] = input_df[feature]
            else:
                input_df_timing[feature] = 0  # Default value for missing features
        
        for feature in depth_feature_names:
            if feature in input_df.columns:
                input_df_depth[feature] = input_df[feature]
            else:
                input_df_depth[feature] = 0  # Default value for missing features
        
        # Make predictions
        timing_violation_prob = timing_violation_model.predict_proba(input_df_timing)[0][1]
        timing_violation = timing_violation_model.predict(input_df_timing)[0]
        combinational_depth = combinational_depth_model.predict(input_df_depth)[0]
        
        # Now that we have the predicted combinational_depth, calculate depth_with_logical_effort
        depth_with_logical_effort = combinational_depth * input_df['logical_effort_estimate'].values[0]
        
        # Recalculate delay_per_logic_level using the predicted combinational_depth
        delay_per_logic_level = input_df['data_path_delay_ps'].values[0] / np.maximum(combinational_depth, 1)
        
        # Enhanced predictions with logical effort
        result = {
            'timing_violation_probability': float(timing_violation_prob),
            'timing_violation_predicted': bool(timing_violation),
            'combinational_depth_predicted': int(combinational_depth),
            'logical_effort_estimate': float(input_df['logical_effort_estimate'].values[0]),
            'depth_with_logical_effort': float(depth_with_logical_effort),
            'estimated_delay_per_level_ps': float(delay_per_logic_level)
        }
        
        return result
    
    return predict_with_pattern_analysis

# Main execution function
def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # 1. Load data
    file_path = 'Cadence_training2.csv'
    df = load_data(file_path)
    
    # 2. Preprocess data with enhanced features
    df_processed = preprocess_data(df)
    
    # Add this debugging code
    print("Columns after preprocessing:", df_processed.columns.tolist())
    
    # 3. Perform exploratory data analysis with new visualizations
    correlation = perform_eda(df_processed)
    
    # 4. Classification model for timing violations
    X_class, y_class = select_features(df_processed, 'classification')
    X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class, test_size=0.2, random_state=42, stratify=y_class)
    
    class_model, class_importance, class_feature_names = build_classification_model(X_train_class, y_train_class, X_test_class, y_test_class)
    
    # 5. Enhanced regression model for combinational depth
    X_reg, y_reg = select_features(df_processed, 'regression')
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    
    reg_model, reg_importance, reg_feature_names = build_regression_model(X_train_reg, y_train_reg, X_test_reg, y_test_reg)
    
    # 6. Create a scalable, deployment-ready predictor with pattern recognition
    predictor = create_scalable_predictor()
    
    # 7. Example prediction with pattern recognition
    example_input = {
        'module_name': 'wallace_tree_multiplier',
        'clock_frequency_mhz': 1000,
        'slack_ps': -11,
        'data_path_delay_ps': 511,
        'required_time_ps': 500,
        'output_delay_ps': 500,
        'timing_violation_risk': 'Low',
        'max_logic_depth': 3,
        'rtl_max_fanout': 8,
        'timing_worst_logic_depth': 1,
        'timing_worst_slack': -11,
        'timing_worst_data_path_delay': 511,
        'timing_avg_logic_depth': 2,
        'cell_count': 34,
        'combinational_logic': 16,
        'arithmetic_ops': 9,
        'multiplexers': 0,
        'logic_and_percent': 14.9,
        'logic_or_percent': 0,
        'logic_not_percent': 0.5,
        'logic_xor_percent': 0,
        'logic_buffer_percent': 25.1,
        'logic_sequential_percent': 59.5,
        'technology': '180',
        'rtl_combinational_block_count': 28,
        'rtl_sequential_block_count': 8
    }
    
    # Sample RTL code for pattern recognition (simplified)
    sample_rtl = """
    module wallace_tree_multiplier (
        input [7:0] a, b,
        output [15:0] out
    );
        // 8 adders in parallel
        assign out = a * b;
    endmodule
    """
    
    # Make prediction with pattern analysis
    prediction = predictor(example_input, sample_rtl)
    
    print(f"Example prediction with pattern recognition:")
    print(f"Timing Violation Probability: {prediction['timing_violation_probability']:.4f}")
    print(f"Timing Violation Predicted: {prediction['timing_violation_predicted']}")
    print(f"Combinational Depth Predicted: {prediction['combinational_depth_predicted']}")
    print(f"Logical Effort Estimate: {prediction['logical_effort_estimate']:.4f}")
    print(f"Depth with Logical Effort: {prediction['depth_with_logical_effort']:.2f}")
    print(f"Estimated Delay per Logic Level: {prediction['estimated_delay_per_level_ps']:.2f} ps")
    
    if 'identified_patterns' in prediction:
        print("\nIdentified patterns:")
        for pattern, count in prediction['identified_patterns'].items():
            print(f"  - {pattern}: {count} instances")
        
        print(f"\nScalability impact: {prediction['estimated_processing_speedup']}")
    
    print("\nComplete! Enhanced models and visualizations have been saved.")

if __name__ == "__main__":
    main()
