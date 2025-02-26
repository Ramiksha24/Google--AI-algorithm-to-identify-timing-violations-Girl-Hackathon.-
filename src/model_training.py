import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

try:
    # Load dataset
    df = pd.read_csv('final_training_cleaned.csv')
    logging.info(f"Dataset loaded. Shape: {df.shape}")

    # Split features and target
    X = df.drop('combinational_depth', axis=1)
    y = df['combinational_depth']

    # Define categorical and numerical columns
    categorical_cols = ['module_type']
    numerical_cols = [
        col for col in X.columns 
        if col not in categorical_cols 
        and col not in ['module_name', 'signal_name']
    ]

    # Remove non-feature columns
    X = X.drop(['module_name', 'signal_name'], axis=1, errors='ignore')

    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # XGBoost model with potential hyperparameter grid
    param_grid = {
        'model__n_estimators': [100, 200, 300],
        'model__learning_rate': [0.01, 0.1, 0.3],
        'model__max_depth': [3, 6, 9]
    }

    # Create XGBoost model
    xgb_model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=42
    )

    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', xgb_model)
    ])

    # Optional: Uncomment for hyperparameter tuning
    # grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_absolute_error')
    # grid_search.fit(X_train, y_train)
    # best_pipeline = grid_search.best_estimator_
    # logging.info(f"Best parameters: {grid_search.best_params_}")

    # Cross-validation
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
    logging.info(f"Cross-validation MAE: {-cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Train model
    pipeline.fit(X_train, y_train)

    # Make predictions
    y_pred = pipeline.predict(X_test)

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    logging.info(f"Mean Absolute Error (MAE): {mae:.3f}")
    logging.info(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
    logging.info(f"R² Score: {r2:.3f}")

    # Save model
    joblib.dump(pipeline, 'xgb_depth_predictor.joblib')
    logging.info("Model saved successfully.")

    # Get feature importance
    feature_names = numerical_cols + list(
        pipeline.named_steps['preprocessor']
        .named_transformers_['cat']
        .get_feature_names_out(categorical_cols)
    )
    feature_importances = pipeline.named_steps['model'].feature_importances_

    # Create DataFrame for feature importance
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    })
    importance_df = importance_df.sort_values('Importance', ascending=False)

    # Plot feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(15))
    plt.title('Top 15 Features for Predicting Combinational Depth')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    logging.info("Feature importance plot saved.")

    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel('Actual Depth')
    plt.ylabel('Predicted Depth')
    plt.title('Actual vs Predicted Combinational Depth')
    plt.tight_layout()
    plt.savefig('actual_vs_predicted.png')
    logging.info("Actual vs Predicted plot saved.")

except Exception as e:
    logging.error(f"An error occurred: {e}")
