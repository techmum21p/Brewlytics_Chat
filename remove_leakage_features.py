"""
Remove data leakage features from processed datasets
"""
import joblib
import numpy as np
import pandas as pd

# Load processed data
processed_dir = 'Cafe_Rewards_Offers/processed'

X_train = joblib.load(f'{processed_dir}/X_train_scaled.pkl')
X_test = joblib.load(f'{processed_dir}/X_test_scaled.pkl')
y_train = joblib.load(f'{processed_dir}/y_train.pkl')
y_test = joblib.load(f'{processed_dir}/y_test.pkl')
feature_names = joblib.load(f'{processed_dir}/feature_names.pkl')

print("="*70)
print("ORIGINAL DATA")
print("="*70)
print(f"Features: {len(feature_names)}")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"\nCurrent features:")
for i, name in enumerate(feature_names, 1):
    print(f"  {i:2}. {name}")

# Identify leakage features
leakage_features = ['offer_completed', 'offer_viewed']  # offer_viewed is also suspicious
present_leakage = [f for f in leakage_features if f in feature_names]

print(f"\n{'='*70}")
print("DATA LEAKAGE DETECTION")
print("="*70)
print(f"\nPotential leakage features found: {present_leakage}")

if 'offer_completed' in feature_names:
    # Check correlation with target
    offer_completed_idx = feature_names.index('offer_completed')
    offer_completed_train = X_train.iloc[:, offer_completed_idx] if isinstance(X_train, pd.DataFrame) else X_train[:, offer_completed_idx]

    if isinstance(offer_completed_train, pd.Series):
        correlation = np.corrcoef(offer_completed_train.values, y_train.values)[0, 1]
    else:
        correlation = np.corrcoef(offer_completed_train, y_train)[0, 1]

    print(f"\n⚠️  CRITICAL: 'offer_completed' correlation with target: {correlation:.4f}")
    if abs(correlation) > 0.9:
        print(f"   This is SEVERE data leakage! (correlation > 0.9)")

# Remove leakage features
features_to_remove = ['offer_completed']  # Start with the most obvious one
print(f"\n{'='*70}")
print("REMOVING LEAKAGE FEATURES")
print("="*70)
print(f"\nFeatures to remove: {features_to_remove}")

# Get indices of features to keep
indices_to_keep = [i for i, f in enumerate(feature_names) if f not in features_to_remove]
features_to_keep = [f for f in feature_names if f not in features_to_remove]

# Remove from datasets
if isinstance(X_train, pd.DataFrame):
    X_train_clean = X_train.drop(columns=features_to_remove)
    X_test_clean = X_test.drop(columns=features_to_remove)
else:
    X_train_clean = X_train[:, indices_to_keep]
    X_test_clean = X_test[:, indices_to_keep]

print(f"\n{'='*70}")
print("CLEANED DATA")
print("="*70)
print(f"Features: {len(features_to_keep)} (removed {len(features_to_remove)})")
print(f"X_train_clean shape: {X_train_clean.shape}")
print(f"X_test_clean shape: {X_test_clean.shape}")

print(f"\nRemaining features:")
for i, name in enumerate(features_to_keep, 1):
    print(f"  {i:2}. {name}")

# Save cleaned data
joblib.dump(X_train_clean, f'{processed_dir}/X_train_scaled.pkl')
joblib.dump(X_test_clean, f'{processed_dir}/X_test_scaled.pkl')
joblib.dump(features_to_keep, f'{processed_dir}/feature_names.pkl')

# Also save backup of original data
import os
backup_dir = f'{processed_dir}/backup_with_leakage'
os.makedirs(backup_dir, exist_ok=True)
joblib.dump(X_train, f'{backup_dir}/X_train_scaled_with_leakage.pkl')
joblib.dump(X_test, f'{backup_dir}/X_test_scaled_with_leakage.pkl')
joblib.dump(feature_names, f'{backup_dir}/feature_names_with_leakage.pkl')

print(f"\n{'='*70}")
print("✓ SAVE COMPLETE")
print("="*70)
print(f"\nCleaned files saved:")
print(f"  - {processed_dir}/X_train_scaled.pkl")
print(f"  - {processed_dir}/X_test_scaled.pkl")
print(f"  - {processed_dir}/feature_names.pkl")
print(f"\nBackup of original files (with leakage):")
print(f"  - {backup_dir}/X_train_scaled_with_leakage.pkl")
print(f"  - {backup_dir}/X_test_scaled_with_leakage.pkl")
print(f"  - {backup_dir}/feature_names_with_leakage.pkl")

print(f"\n{'='*70}")
print("NEXT STEPS")
print("="*70)
print(f"\n1. Re-run modeling notebooks with cleaned data")
print(f"2. Expect lower (but realistic) performance metrics")
print(f"3. Re-run PCA analysis")
print(f"4. Re-run SHAP analysis")
print(f"5. Update all summaries and conclusions")
