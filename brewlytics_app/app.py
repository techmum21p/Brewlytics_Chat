import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from io import BytesIO
from google import genai
from openai import OpenAI
import anthropic

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

@st.cache_data
def load_segment_data():
    customers_clusters_path = 'Cafe_Rewards_Offers/segmentation/customers_with_clusters.csv'
    segment_profiles_path = 'Cafe_Rewards_Offers/segmentation/segment_profiles.csv'
    
    if os.path.exists(customers_clusters_path) and os.path.exists(segment_profiles_path):
        customers_clusters = pd.read_csv(customers_clusters_path)
        segment_profiles = pd.read_csv(segment_profiles_path)
        return customers_clusters, segment_profiles
    return None, None

def match_customer_to_cluster(customer_row, customers_clusters):
    if customers_clusters is None:
        return None, "New Customer"
    
    matching_rows = customers_clusters[
        (customers_clusters['age'] == customer_row['age']) &
        (customers_clusters['income'] == customer_row['income']) &
        (customers_clusters['gender'] == customer_row['gender']) &
        (customers_clusters['age_group'] == customer_row['age_group']) &
        (customers_clusters['income_bracket'] == customer_row['income_bracket']) &
        (customers_clusters['tenure_group'] == customer_row['tenure_group'])
    ]
    
    if len(matching_rows) > 0:
        cluster_id = matching_rows.iloc[0]['cluster']
        return int(cluster_id), f"Cluster {cluster_id}"
    
    return None, "New Customer"

def get_segment_profile(cluster_id, segment_profiles):
    if cluster_id is None or segment_profiles is None:
        return {
            'name': 'New Customer',
            'strategy': 'Cold Start - Welcome and engagement strategy needed',
            'performance_tier': 'NEW',
            'completion_rate': 0.0,
            'description': 'New customer with no historical data'
        }
    
    cluster_row = segment_profiles[segment_profiles['cluster'] == cluster_id]
    if len(cluster_row) > 0:
        row = cluster_row.iloc[0]
        return {
            'name': row['name'],
            'strategy': row['strategy'],
            'performance_tier': row['performance_tier'],
            'completion_rate': row['completion_rate'],
            'description': f"{row['performance_tier']} segment with {row['completion_rate']*100:.1f}% completion rate"
        }
    
    return {
        'name': 'New Customer',
        'strategy': 'Cold Start - Welcome and engagement strategy needed',
        'performance_tier': 'NEW',
        'completion_rate': 0.0,
        'description': 'New customer with no historical data'
    }

def build_strategy_prompt(customer_data, segment_profile):
    """Build a concise prompt for marketing strategy generation."""
    return f"""Customer: {customer_data.get('age_group', 'N/A')} age, {customer_data.get('income_bracket', 'N/A')} income, {customer_data.get('tenure_group', 'N/A')} tenure
Offer: {customer_data.get('offer_type', 'N/A')}, difficulty {customer_data.get('difficulty', 'N/A')}, {customer_data.get('duration', 'N/A')} days
Segment: {segment_profile['name']} ({segment_profile['performance_tier']})
Completion probability: {customer_data.get('completion_probability', 0)*100:.0f}%

Give exactly 3 bullet-point marketing strategies to help this customer complete the offer. Be specific and actionable. Format:
• Strategy 1
• Strategy 2
• Strategy 3"""

def generate_marketing_strategy_gemini(customer_data, segment_profile, api_key, model="gemini-2.5-flash"):
    try:
        client = genai.Client(api_key=api_key)
        prompt = build_strategy_prompt(customer_data, segment_profile)
        response = client.models.generate_content(model=model, contents=prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def generate_marketing_strategy_openai(customer_data, segment_profile, api_key, model="gpt-4.1-mini"):
    try:
        client = OpenAI(api_key=api_key)
        prompt = build_strategy_prompt(customer_data, segment_profile)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def generate_marketing_strategy_anthropic(customer_data, segment_profile, api_key, model="claude-haiku-4-5"):
    try:
        client = anthropic.Anthropic(api_key=api_key)
        prompt = build_strategy_prompt(customer_data, segment_profile)
        response = client.messages.create(
            model=model,
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()
    except Exception as e:
        return f"Error: {str(e)}"

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
    
    with st.sidebar:
        st.header("🔑 LLM Configuration")
        st.markdown("Configure your LLM provider for generating personalized marketing strategies.")
        
        llm_provider = st.selectbox(
            "Select LLM Provider",
            options=["Google Gemini", "OpenAI", "Anthropic Claude"],
            index=0
        )
        
        api_key = st.text_input(
            "API Key",
            type="password",
            placeholder="Enter your API key...",
            help="Your API key will only be used for the current session and not stored."
        )
        
        if llm_provider == "Google Gemini":
            model_name = st.selectbox(
                "Model",
                options=["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro"],
                index=0
            )
        elif llm_provider == "OpenAI":
            model_name = st.selectbox(
                "Model",
                options=["gpt-4.1-mini", "gpt-4.1", "gpt-4o-mini", "gpt-4o"],
                index=0
            )
        else:
            model_name = st.selectbox(
                "Model",
                options=["claude-haiku-4-5", "claude-sonnet-4-5"],
                index=0
            )
        
        st.markdown("---")
        st.info("💡 Configure your LLM above to enable personalized marketing strategy generation after predictions.")
    
    model, scaler, feature_names = load_model_and_artifacts()
    customers_clusters, segment_profiles = load_segment_data()
    
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

                        st.session_state['result_df'] = result_df
                        st.session_state['predictions_generated'] = True

                    except Exception as e:
                        st.error(f"❌ Error during prediction: {str(e)}")
                        st.error("Please check that your CSV contains all required columns.")
                        st.write("Required columns include: received_time, difficulty, duration, age, income, offer_type, gender, etc.")

            # Display results if predictions have been generated (persists across reruns)
            if st.session_state.get('predictions_generated') and 'result_df' in st.session_state:
                result_df = st.session_state['result_df']

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
                    st.dataframe(result_df.head(50), width="stretch")

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
                    st.dataframe(top_customers, width="stretch")

                st.markdown("---")
                st.markdown("### 🎯 Dynamic Marketing Strategy Generator")

                if not api_key:
                    st.warning("⚠️ Please configure your LLM API key in the sidebar to generate marketing strategies.")
                else:
                    st.info(f"✅ LLM configured: {llm_provider} - {model_name}")

                    st.caption("💡 **Free tier limits:** Gemini ~15 RPM, OpenAI GPT-4o ~3 RPM. Enable rate limiting to process more customers safely.")

                    col_strategy1, col_strategy2 = st.columns([1, 1])

                    with col_strategy1:
                        num_strategies = st.number_input(
                            "Number of customers",
                            min_value=1,
                            max_value=100,
                            value=10,
                            key="num_strategies_input",
                            help="Generate strategies for top N customers."
                        )

                    with col_strategy2:
                        filter_by = st.selectbox(
                            "Generate strategies for",
                            options=["Most Likely to Complete", "Least Likely to Complete", "All Customers"],
                            index=0,
                            key="filter_by_input"
                        )

                    # Rate limiting toggle
                    rate_limit_enabled = st.checkbox(
                        "Enable rate limiting (recommended for free tiers)",
                        value=True,
                        key="rate_limit_checkbox",
                        help="Adds delays between API calls to stay within free tier limits"
                    )

                    if rate_limit_enabled:
                        st.caption("⏱️ **Auto-calculated delays:** Gemini = 4s/request (15 RPM), OpenAI = 20s/request (3 RPM), Anthropic = 12s/request (5 RPM)")

                    if st.button("🚀 Generate Marketing Strategies", type="primary", key="generate_strategies_btn"):
                        import time

                        # Rate limits (seconds per request to stay under RPM limits)
                        rate_limits = {
                            "Google Gemini": 4,      # 15 RPM = 4 seconds per request
                            "OpenAI": 20,            # 3 RPM = 20 seconds per request
                            "Anthropic Claude": 12   # 5 RPM = 12 seconds per request
                        }
                        delay_per_request = rate_limits.get(llm_provider, 10) if rate_limit_enabled else 0

                        # Calculate estimated time
                        if rate_limit_enabled:
                            est_time = num_strategies * delay_per_request
                            est_minutes = est_time // 60
                            est_seconds = est_time % 60
                            time_str = f"{int(est_minutes)}m {int(est_seconds)}s" if est_minutes > 0 else f"{int(est_seconds)}s"
                            st.info(f"⏱️ Estimated time with rate limiting: ~{time_str}")

                        with st.spinner("Generating personalized marketing strategies..."):
                            try:
                                if filter_by == "Most Likely to Complete":
                                    target_df = st.session_state['result_df'].nlargest(num_strategies, 'completion_probability')
                                elif filter_by == "Least Likely to Complete":
                                    target_df = st.session_state['result_df'].nsmallest(num_strategies, 'completion_probability')
                                else:
                                    target_df = st.session_state['result_df'].head(num_strategies)

                                target_df = target_df.copy()
                                target_df['segment'] = 'Unknown'
                                target_df['recommended_strategy'] = 'Pending...'

                                progress_bar = st.progress(0)
                                status_text = st.empty()

                                for idx, (i, row) in enumerate(target_df.iterrows()):
                                    if rate_limit_enabled and delay_per_request > 0:
                                        remaining = (len(target_df) - idx) * delay_per_request
                                        rem_min = remaining // 60
                                        rem_sec = remaining % 60
                                        time_remaining = f"{int(rem_min)}m {int(rem_sec)}s" if rem_min > 0 else f"{int(rem_sec)}s"
                                        status_text.text(f"Processing customer {idx + 1} of {len(target_df)}... (~{time_remaining} remaining)")
                                    else:
                                        status_text.text(f"Processing customer {idx + 1} of {len(target_df)}...")

                                    customer_data = row.to_dict()

                                    cluster_id, segment_name = match_customer_to_cluster(customer_data, customers_clusters)
                                    segment_profile = get_segment_profile(cluster_id, segment_profiles)

                                    target_df.at[i, 'segment'] = segment_profile['name']

                                    if llm_provider == "Google Gemini":
                                        strategy = generate_marketing_strategy_gemini(
                                            customer_data, segment_profile, api_key, model_name
                                        )
                                    elif llm_provider == "OpenAI":
                                        strategy = generate_marketing_strategy_openai(
                                            customer_data, segment_profile, api_key, model_name
                                        )
                                    else:
                                        strategy = generate_marketing_strategy_anthropic(
                                            customer_data, segment_profile, api_key, model_name
                                        )

                                    target_df.at[i, 'recommended_strategy'] = strategy

                                    progress = (idx + 1) / len(target_df)
                                    progress_bar.progress(progress)

                                    # Apply rate limiting delay (skip on last item)
                                    if rate_limit_enabled and idx < len(target_df) - 1:
                                        time.sleep(delay_per_request)

                                progress_bar.progress(1.0)
                                status_text.text("✅ Marketing strategies generated successfully!")

                                # Store strategies in session state
                                st.session_state['strategies_df'] = target_df
                                st.session_state['strategies_generated'] = True

                            except Exception as e:
                                st.error(f"❌ Error generating strategies: {str(e)}")
                                st.error("Please check your API key and try again.")

                    # Display strategies if they have been generated
                    if st.session_state.get('strategies_generated') and 'strategies_df' in st.session_state:
                        target_df = st.session_state['strategies_df']

                        st.markdown("### 📋 Marketing Strategies Results")
                        st.dataframe(
                            target_df[[
                                'age', 'income', 'gender', 'tenure_group', 'offer_type',
                                'completion_probability', 'segment', 'recommended_strategy'
                            ]],
                            width="stretch",
                            column_config={
                                "recommended_strategy": st.column_config.TextColumn(
                                    "Recommended Strategy",
                                    width="large"
                                )
                            }
                        )

                        strategies_csv = target_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Marketing Strategies (CSV)",
                            data=strategies_csv,
                            file_name=f"brewlytics_marketing_strategies_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            key="download_strategies_btn"
                        )
        
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
