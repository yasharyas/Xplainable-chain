"""
Simple ML Model Training Script for XAI-Chain
Trains an XGBoost model on synthetic transaction data
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
import pickle
import os

def train_model():
    """Train XGBoost model on transaction data"""
    
    print("Starting XAI-Chain Model Training...")
    print("=" * 60)
    
    # check if data file is there
    data_path = 'data/sample_transactions.csv'
    if not os.path.exists(data_path):
        print("ERROR: Training data not found!")
        print("Please run: python data/generate_sample_data.py")
        return
    
    # load the data from csv
    print("\nLoading data...")
    df = pd.read_csv(data_path)
    print(f"   Total transactions: {len(df)}")
    print(f"   Malicious: {df['is_malicious'].sum()} ({df['is_malicious'].mean()*100:.1f}%)")
    print(f"   Normal: {(df['is_malicious']==0).sum()} ({(df['is_malicious']==0).mean()*100:.1f}%)")
    
    # get the features ready for training
    print("\nPreparing features...")
    feature_columns = [
        'amount', 'gas_price', 'gas_used', 'num_transfers',
        'unique_addresses', 'time_of_day', 'contract_interaction',
        'sender_tx_count', 'receiver_tx_count'
    ]
    
    X = df[feature_columns]
    y = df['is_malicious']
    
    print(f"   Features: {X.shape[1]}")
    print(f"   Feature names: {', '.join(feature_columns)}")
    
    # split data into training and testing
    print("\nSplitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Training set: {len(X_train)} samples")
    print(f"   Test set: {len(X_test)} samples")
    
    # scale the features so they are all similar size
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # train the machine learning model
    print("\nTraining XGBoost model...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    
    model.fit(
        X_train_scaled, 
        y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=False
    )
    
    # test how good the model is
    print("\nEvaluating model...")
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    # show the results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Malicious']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"   True Negatives:  {cm[0][0]:5d}")
    print(f"   False Positives: {cm[0][1]:5d}")
    print(f"   False Negatives: {cm[1][0]:5d}")
    print(f"   True Positives:  {cm[1][1]:5d}")
    
    print(f"\nROC AUC Score: {roc_auc_score(y_test, y_pred_proba[:, 1]):.4f}")
    
    # show which features matter most
    print("\nTop 5 Most Important Features:")
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in feature_importance.head(5).iterrows():
        print(f"   {row['feature']:30s}: {row['importance']:.4f}")
    
    # save the trained model to disk
    print("\nSaving model...")
    os.makedirs('app/ml', exist_ok=True)
    
    model_path = 'app/ml/model.pkl'
    scaler_path = 'app/ml/scaler.pkl'
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"   Model saved to: {model_path}")
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"   Scaler saved to: {scaler_path}")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Start the backend: uvicorn app.main:app --reload")
    print("2. Test the API: http://localhost:8000/docs")
    print("3. Start the frontend: cd frontend && npm run dev")

if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        print(f"\nERROR during training: {e}")
        print("\nPlease ensure:")
        print("- You have run: python data/generate_sample_data.py")
        print("- All required packages are installed: pip install -r requirements.txt")
        import traceback
        traceback.print_exc()
