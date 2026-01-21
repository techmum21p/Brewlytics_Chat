# ============================================================================
# ADD THIS TO YOUR 03_MODELING.IPYNB
# ============================================================================

# INSTRUCTIONS:
# 1. Copy this entire code block
# 2. Create a NEW code cell in your notebook
# 3. Paste this code into that new cell
# 4. Place it RIGHT AFTER the "Load Processed Data" cell (after loading data)

# ============================================================================
# REMOVE DATA LEAKAGE
# ============================================================================

print("=" * 60)
print("CHECKING FOR DATA LEAKAGE")
print("=" * 60)

# Check if offer_completed is in features (THIS IS THE PROBLEM!)
if 'offer_completed' in X_train.columns:
    print("\n⚠️  DATA LEAKAGE DETECTED!")
    print("Column 'offer_completed' found in features.")
    print("This is IDENTICAL to target, causing perfect 1.0 predictions.")
    print("\nDropping 'offer_completed' from train and test sets...")
    
    X_train.drop('offer_completed', axis=1, inplace=True)
    X_test.drop('offer_completed', axis=1, inplace=True)
    
    # Update global feature_names
    global feature_names
    feature_names = [f for f in feature_names if f != 'offer_completed']
    
    print(f"✓ Dropped. New shape: {X_train.shape}")
    print(f"✓ Features remaining: {len(feature_names)}")
else:
    print("\n✓ No 'offer_completed' column found (already removed)")

# Check for offer_viewed (less severe leak, but still worth noting)
if 'offer_viewed' in X_train.columns:
    print("\nℹ️  INFO: 'offer_viewed' feature present")
    print("=" * 60)
    print("This feature is a potential data leak.")
    print("It's available BEFORE completion in real-time scenarios.")
    print("=" * 60)
    print("\nRECOMMENDATION:")
    print("For true real-time prediction models, consider:")
    print("  1. Train models WITHOUT 'offer_viewed'")
    print("  2. For 'post-notification' prediction, keep it")
    print("\nFor now, we'll keep it to see model performance.")
else:
    print("✓ No 'offer_viewed' feature found")

# Final verification
print("\n" + "=" * 60)
print("FINAL DATA SHAPE")
print("=" * 60)
print(f"Train: {X_train.shape[0]:,} samples × {X_train.shape[1]} features")
print(f"Test:  {X_test.shape[0]:,} samples × {X_test.shape[1]} features")
print(f"Features: {len(feature_names)}")
