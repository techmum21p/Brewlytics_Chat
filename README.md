# Brewlytics Chat: Starbucks Customer Analytics & Offer Completion Prediction

## 📊 Project Overview

Brewlytics Chat is a comprehensive data science project that analyzes Starbucks customer behavior and builds machine learning models to predict offer completion rates. The project demonstrates the complete data science pipeline from exploratory data analysis to model deployment and interpretation.

## 🎯 Business Objective

Predict which customers are likely to complete promotional offers to optimize marketing strategies, improve customer engagement, and maximize ROI on promotional campaigns.

## 📁 Project Structure

```
Brewlytics_Chat/
├── Notebooks/                    # Jupyter Notebooks (01-06)
│   ├── 01_EDA.ipynb            # Exploratory Data Analysis
│   ├── 02_Feature_Engg.ipynb    # Feature Engineering
│   ├── 03_Modeling.ipynb       # Machine Learning Models
│   ├── 04_PCA.ipynb            # Dimensionality Reduction
│   ├── 05_SHAP.ipynb           # Model Explainability
│   ├── 06_Customer_Segmentation.ipynb # Customer Segmentation
│   └── 07_Bias_Fairness_Analysis.ipynb # Bias & Fairness Analysis
├── Cafe_Rewards_Offers/         # Dataset & Processed Data
│   ├── customers.csv           # Customer demographics
│   ├── offers.csv              # Promotional offer details
│   ├── events.csv              # Transaction events
│   ├── processed/             # Processed datasets for ML
│   └── models/                 # Trained machine learning models
└── README.md                    # This file
```

## 📈 Dataset Overview

The project uses three main datasets:
- **Customers**: 17,000 records with demographic information (age, gender, income, membership details)
- **Offers**: 10 types of promotional offers (bogo, discount, informational) with varying difficulty and duration
- **Events**: 306,534 transaction events tracking offer reception, viewing, and completion

## 🔍 Key Findings from Analysis

### 1. Customer Demographics (Notebook 01)
- **Gender Distribution**: 57.2% male, 41.3% female, 1.4% other
- **Age Profile**: Middle-aged customers (40-65 years) form the core demographic
- **Income Distribution**: Middle-class customers ($50K-$75K) with multimodal patterns
- **Membership Growth**: Explosive growth in 2017 (6,500+ new members), indicating successful market penetration
- **Data Quality**: 12.8% of customers missing demographics (MNAR - Missing Not At Random)

### 2. Feature Engineering (Notebook 02)
- **Data Leakage Removal**: Critical step removing `offer_completed` and `offer_viewed` features
- **Final Feature Set**: 24 clean features available at prediction time
- **Encoding Strategy**: 
  - One-hot encoding for nominal variables (offer_type, gender)
  - Ordinal encoding for ordered variables (age_group, income_bracket, tenure_group)
- **Missing Value Handling**: Median imputation for 0.13% missing values
- **Scaling**: StandardScaler applied to 11 numerical features

### 3. Machine Learning Models (Notebook 03)
- **Best Model**: Random Forest (F1 = 0.8601, AUC-ROC = 0.9277)
- **Model Comparison**:
  - Logistic Regression: F1 = 0.8240 (baseline)
  - Decision Tree: F1 = 0.8263
  - Random Forest: F1 = 0.8601 (winner)
  - XGBoost: F1 = 0.8515
- **Key Insight**: Ensemble methods significantly outperform simple baselines

### 4. Dimensionality Reduction (Notebook 04)
- **PCA Analysis**: 8 components capture 90% variance with 67% feature reduction
- **Performance Trade-off**: Only 1.30% F1 drop for 67% fewer features
- **Recommendation**: Use 8-component PCA model for production (efficiency gains)
- **Feature Contributions**: 
  - PC1: Customer tenure (membership_year, duration_days)
  - PC2: Demographics (age, income)
  - PC3: Offer characteristics (duration, difficulty)

### 5. Model Explainability (Notebook 05)
- **SHAP Analysis**: Revealed actual feature importance and directional impacts
- **Top Features**:
  1. `offer_type_discount` (21.41% importance) - Strongest completion driver
  2. `duration` (14.16% importance) - Shorter duration = higher completion
  3. `difficulty` (9.27% importance) - Easier offers = higher completion
- **Key Insight**: Offer design matters more than customer demographics (52% vs 34% importance)

### 6. Customer Segmentation (Notebook 06)
- **Optimal Segments**: 5 distinct customer clusters identified
- **Segment Profiles**:
  1. **New Male Members** (30.8%): Age 52, $60K income, 0.7 years tenure, 44.6% completion
  2. **Affluent Female Members** (36.2%): Age 58, $72K income, 1.3 years tenure, 65.9% completion
  3. **Missing Demographics** (11.5%): Data quality issue, 15.7% completion
  4. **Small Engaged Segment** (1.2%): High engagement, 63.2% completion
  5. **Long-Tenure Male Members** (20.2%): Age 53, $63K income, 2.9 years tenure, 65.3% completion

## 🚀 Technical Implementation

### Data Pipeline
1. **Raw Data** → **Quality Checks** → **Feature Engineering** → **Model Training** → **Evaluation** → **Interpretation**

### Key Technologies
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Machine Learning**: Random Forest, XGBoost, Logistic Regression
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Interpretability**: SHAP, PCA
- **Clustering**: K-means

### Model Performance
- **Accuracy**: 84.54%
- **Precision**: 83.18%
- **Recall**: 89.05%
- **F1-Score**: 86.01%
- **AUC-ROC**: 92.77%

## 💡 Business Recommendations

### 1. Offer Design Optimization (Highest Priority)
- **Increase discount offers** - 21.41% importance, strongest completion driver
- **Reduce offer duration** - Negative impact when too long
- **Lower difficulty thresholds** - Easier offers perform better

### 2. Customer Targeting Strategy
- **Primary Focus**: Affluent female members (36.2% of base, 65.9% completion)
- **Secondary Focus**: Long-tenure male members (20.2% of base, 65.3% completion)
- **Growth Opportunity**: New male members (30.8% of base, need engagement boost)

### 3. Channel Optimization
- Prioritize channels with highest engagement rates
- Focus on improving offer visibility (view rate strongly correlates with completion)

### 4. Data Quality Improvement
- **Critical**: Fix missing demographics collection (11.5% of customers)
- Implement better onboarding data capture
- Regular data quality audits

## 📊 Model Insights

### Feature Importance (Top 10)
1. `membership_duration_days` (17.35%)
2. `income` (13.08%)
3. `age` (11.34%)
4. `offer_type_informational` (8.77%)
5. `difficulty` (8.37%)
6. `duration` (8.04%)
7. `received_time` (5.52%)
8. `membership_month` (5.48%)
9. `income_bracket_encoded` (3.27%)
10. `age_group_encoded` (3.08%)

### SHAP Directional Impacts
- **Positive for Completion**: Discount offers, BOGO offers, higher income
- **Negative for Completion**: Longer duration, higher difficulty, informational offers

## 🎯 Use Cases

### 1. Real-time Offer Recommendation
- Deploy Random Forest model with 24 features
- Predict completion probability at offer delivery
- Route high-probability offers to appropriate customers

### 2. Customer Segmentation Marketing
- Use 5-cluster segmentation for targeted campaigns
- Tailor offer types and messaging to each segment
- Monitor segment performance over time

### 3. Offer Design Optimization
- Use SHAP insights to design better offers
- A/B test new offer configurations
- Optimize difficulty and duration parameters

### 4. Customer Lifetime Value Prediction
- Leverage membership duration and engagement patterns
- Identify high-value customers for retention programs
- Predict churn risk based on behavior changes

## 🔧 Model Deployment

### Production Options
1. **Full Feature Model**: Maximum accuracy (F1 = 0.8601)
2. **PCA Reduced Model**: 67% faster with minimal accuracy loss (F1 = 0.8472)

### Monitoring Requirements
- Feature drift detection (monthly)
- Performance degradation alerts
- Segment size changes
- Data quality metrics

## 📚 Learning Outcomes

This project demonstrates:
- **Complete ML Pipeline**: From raw data to production-ready models
- **Advanced Techniques**: PCA, SHAP, clustering, ensemble methods
- **Business Acumen**: Translating technical insights into actionable recommendations
- **Data Quality Management**: Handling missing data and preventing leakage
- **Model Interpretability**: Explaining black-box models for stakeholder trust

## 🛠️ How to Run

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn shap xgboost jupyter
```

### Execution Order
1. Run `01_EDA.ipynb` - Understand the data
2. Run `02_Feature_Engg.ipynb` - Prepare features for modeling
3. Run `03_Modeling.ipynb` - Train and evaluate models
4. Run `04_PCA.ipynb` - Optimize feature space
5. Run `05_SHAP.ipynb` - Understand model decisions
6. Run `06_Customer_Segmentation.ipynb` - Discover customer segments

### Data Files
Ensure the following files are in `Cafe_Rewards_Offers/`:
- `customers.csv`
- `offers.csv` 
- `events.csv`

## 📄 License

This project is for educational purposes. Please ensure compliance with data usage policies and regulations.


---

**Project Status**: ✅ Complete (All notebooks functional and analyzed)
**Last Updated**: January 2025
**Total Analysis**: 6 comprehensive notebooks covering end-to-end data science pipeline