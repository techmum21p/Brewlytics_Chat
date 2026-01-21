# Add this cell to your 03_Modeling.ipynb
# Insert it RIGHT AFTER the "Load Processed Data" cell

# ============================================================================
# REMOVE DATA LEAKAGE
# ============================================================================

# Check if offer_completed is in features and remove it
if 'offer_completed' in X_train.columns:
    print("=" * 60)
    print("⚠️  DATA LEAKAGE DETECTED!")
    print("=" * 60)
    print("Column 'offer_completed' found in features.")
    print("This is IDENTICAL to target, causing perfect predictions.")
    print("\nDropping 'offer_completed' from train and test sets...\n")
    
    X_train = X_train.drop('offer_completed', axis=1)
    X_test = X_test.drop('offer_completed', axis=1)
    
    # Update feature names list
    global feature_names
    feature_names = [f for f in feature_names if f != 'offer_completed']
    
    print(f"✓ Dropped. New shape: {X_train.shape}")
    print(f"✓ Features remaining: {len(feature_names)}")
else:
    print("✓ No data leakage columns detected")

# Also check for offer_viewed (less severe leak)
if 'offer_viewed' in X_train.columns:
    print("\n" + "=" * 60)
    print("ℹ️  INFO: 'offer_viewed' feature present")
    print("=" * 60)
    print("This feature may be considered data leakage.")
    print("We're keeping it for now, but consider removing it")
    print("for a true real-time prediction model.")
    print("\nTo remove, add this line:")
    print("X_train = X_train.drop('offer_viewed', axis=1)")
    print("X_test = X_test.drop('offer_viewed', axis=1)")
