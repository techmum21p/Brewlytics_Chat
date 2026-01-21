import joblib
import pandas as pd
import numpy as np
import streamlit as st
from io import BytesIO
import os

@st.cache_resource
def load_model_and_artifacts():
    model_path = 'Cafe_Rewards_Offers/models/random_forest.pkl'
    scaler_path = 'Cafe_Rewards_Offers/processed/scaler.pkl'
    feature_names_path = 'Cafe_Rewards_Offers/processed/feature_names.pkl'
    
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        return None, None, None
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_names = joblib.load(feature_names_path)
    
    return model, scaler, feature_names

def preprocess_data(df, scaler, feature_names):
    df_processed = df.copy()
    
    categorical_mappings = {
        'offer_type': ['bogo', 'discount', 'informational'],
        'gender': ['F', 'M', 'O', 'Missing'],
        'age_group': ['18-30', '31-45', '46-60', '61-75', '76+'],
        'income_bracket': ['Missing', 'Low', 'Medium', 'High', 'Very High'],
        'tenure_group': ['0-6 months', '6-12 months', '1-2 years', '2+ years']
    }
    
    if 'offer_type' in df_processed.columns:
        dummies = pd.get_dummies(df_processed['offer_type'], prefix='offer_type')
        for col in categorical_mappings['offer_type']:
            col_name = f'offer_type_{col}'
            if col_name not in dummies.columns:
                dummies[col_name] = 0
        df_processed = pd.concat([df_processed.drop('offer_type', axis=1), dummies], axis=1)
    
    if 'gender' in df_processed.columns:
        dummies = pd.get_dummies(df_processed['gender'], prefix='gender')
        for col in categorical_mappings['gender']:
            col_name = f'gender_{col}'
            if col_name not in dummies.columns:
                dummies[col_name] = 0
        df_processed = pd.concat([df_processed.drop('gender', axis=1), dummies], axis=1)
    
    ordinal_mappings = {
        'age_group': {cat: i for i, cat in enumerate(categorical_mappings['age_group'])},
        'income_bracket': {cat: i for i, cat in enumerate(categorical_mappings['income_bracket'])},
        'tenure_group': {cat: i for i, cat in enumerate(categorical_mappings['tenure_group'])}
    }
    
    for col, mapping in ordinal_mappings.items():
        if col in df_processed.columns:
            df_processed[f'{col}_encoded'] = df_processed[col].map(mapping)
        else:
            df_processed[f'{col}_encoded'] = 0
        if col in df_processed.columns:
            df_processed = df_processed.drop(col, axis=1)
    
    cols_to_drop = ['index', 'customer_id', 'offer_id', 'completion_time', 
                   'time_to_action', 'offer_completed', 'offer_viewed', 
                   'became_member_on', 'became_member_date', 'target']
    
    for col in cols_to_drop:
        if col in df_processed.columns:
            df_processed = df_processed.drop(col, axis=1)
    
    for col in df_processed.columns:
        if df_processed[col].dtype == 'object':
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
    
    for col in df_processed.columns:
        if df_processed[col].isnull().any():
            if df_processed[col].dtype in ['int64', 'float64']:
                median_val = df_processed[col].median()
                df_processed[col] = df_processed[col].fillna(median_val)
            else:
                mode_val = df_processed[col].mode()[0] if len(df_processed[col].mode()) > 0 else 0
                df_processed[col] = df_processed[col].fillna(mode_val)
    
    scaler_feature_names = scaler.feature_names_in_
    for col in scaler_feature_names:
        if col not in df_processed.columns:
            df_processed[col] = 0
    
    df_processed[scaler_feature_names] = scaler.transform(df_processed[scaler_feature_names])
    
    missing_features = set(feature_names) - set(df_processed.columns)
    for feat in missing_features:
        df_processed[feat] = 0
    
    df_final = df_processed[feature_names]
    
    return df_final

def main():
    st.set_page_config(
        page_title="Brewlytics - Offer Completion Predictor",
        page_icon="☕",
        layout="wide"
    )
    
    st.title("☕ Brewlytics - Offer Completion Predictor")
    st.markdown("---")
    
    st.markdown("""
    ### Predict which customers will complete offers
    
    Upload your customer dataset (CSV format) containing:
    - Offer details (difficulty, duration, offer type)
    - Customer demographics (age, income, gender)
    - Membership information (duration, tenure group)
    - Marketing channels received
    
    The app will use the trained Random Forest model to predict offer completion likelihood.
    """)
    
    model, scaler, feature_names = load_model_and_artifacts()
    
    if model is None:
        st.stop()
    
    st.markdown("### 📤 Upload Your Dataset")
    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=['csv'],
        help="Upload a CSV file with customer and offer data"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! {len(df)} rows loaded.")
            
            with st.expander("🔍 View Uploaded Data", expanded=False):
                st.dataframe(df.head(10))
                st.write(f"**Total rows:** {len(df):,}")
                st.write(f"**Total columns:** {len(df.columns)}")
                st.write(f"**Columns:** {', '.join(df.columns.tolist())}")
            
            if st.button("🚀 Generate Predictions", type="primary"):
                with st.spinner("Processing data and generating predictions..."):
                    try:
                        df_processed = preprocess_data(df.copy(), scaler, feature_names)
                        
                        predictions = model.predict(df_processed)
                        probabilities = model.predict_proba(df_processed)
                        
                        result_df = df.copy()
                        result_df['prediction'] = predictions
                        result_df['prediction_label'] = result_df['prediction'].map({0: 'Will Not Complete', 1: 'Will Complete'})
                        result_df['completion_probability'] = probabilities[:, 1]
                        result_df['non_completion_probability'] = probabilities[:, 0]
                        
                        st.markdown("---")
                        st.markdown("### 📊 Prediction Results")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            total_predictions = len(result_df)
                            st.metric("Total Predictions", f"{total_predictions:,}")
                        
                        with col2:
                            will_complete = (result_df['prediction'] == 1).sum()
                            completion_rate = (will_complete / total_predictions * 100)
                            st.metric("Will Complete Offer", f"{will_complete:,} ({completion_rate:.1f}%)")
                        
                        with col3:
                            will_not_complete = (result_df['prediction'] == 0).sum()
                            non_completion_rate = (will_not_complete / total_predictions * 100)
                            st.metric("Will Not Complete", f"{will_not_complete:,} ({non_completion_rate:.1f}%)")
                        
                        st.markdown("---")
                        
                        col_left, col_right = st.columns([2, 1])
                        
                        with col_left:
                            st.markdown("#### 📋 Detailed Predictions")
                            st.dataframe(result_df.head(50), use_container_width=True)
                            
                            csv = result_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Full Results (CSV)",
                                data=csv,
                                file_name=f"brewlytics_predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                        
                        with col_right:
                            st.markdown("#### 📈 Probability Distribution")
                            
                            prob_bins = pd.cut(result_df['completion_probability'], 
                                            bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                            labels=['0-20%', '20-40%', '40-60%', '60-80%', '80-100%'])
                            prob_counts = prob_bins.value_counts().sort_index()
                            
                            st.bar_chart(prob_counts)
                            
                            st.markdown("#### 🔝 Top 10 Most Likely to Complete")
                            top_customers = result_df.nlargest(10, 'completion_probability')[['age', 'income', 'completion_probability']]
                            st.dataframe(top_customers, use_container_width=True)
                    
                    except Exception as e:
                        st.error(f"❌ Error during prediction: {str(e)}")
                        st.error("Please check that your CSV contains all required columns.")
                        st.write("Required columns include: received_time, difficulty, duration, age, income, offer_type, gender, etc.")
        
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
    
    st.markdown("---")
    st.markdown("""
    ### 📋 Required Columns
    
    The CSV should contain the following columns:
    
    **Offer Details:**
    - `received_time`, `difficulty`, `duration`, `offer_type`
    
    **Marketing Channels:**
    - `in_email`, `in_mobile`, `in_social`, `in_web`, `offer_received`
    
    **Customer Demographics:**
    - `age`, `income`, `gender`, `age_group`, `income_bracket`
    
    **Membership Information:**
    - `membership_year`, `membership_duration_days`, `membership_month`, `tenure_group`
    
    **Flags:**
    - `is_demographics_missing`
    
    *Note: The model will automatically drop any leakage columns (offer_completed, offer_viewed, completion_time, etc.)*
    """)

if __name__ == "__main__":
    main()
