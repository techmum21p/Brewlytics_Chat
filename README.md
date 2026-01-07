# Brewlytics: Predictive Offer Completion & Customer Segmentation

A comprehensive machine learning system that predicts which customers will complete digital marketing offers and delivers personalized recommendations to maximize completion rates for a coffee chain's rewards program.

## Project Overview

**Business Problem:** Improve the targeting of promotional offers (Buy-One-Get-One, discounts, and informational campaigns) by predicting which customers are most likely to complete each offer type.

**Solution:** Machine learning pipeline that:
- Predicts offer completion with **86% F1-score** and **93% AUC-ROC**
- Segments customers into 3 actionable groups
- Provides explainable predictions using SHAP analysis
- Ensures fairness across demographic groups
- Optimizes for production deployment with PCA dimensionality reduction

## Key Results

- **Model Performance:** Random Forest classifier achieving 84.5% accuracy, 89.1% recall
- **Customer Segments:**
  - Elite Performers (21%): 81.9% completion rate
  - Moderate Performers (66%): 44.5% completion rate
  - At-Risk (13%): 14.1% completion rate
- **Feature Efficiency:** 67% reduction in features (25→8) with only 1.3% performance loss
- **Business Impact:** Data-driven targeting to increase baseline 53% completion rate

## Data Sources

The project uses three core datasets from the cafe rewards program:

- **customers.csv** - 16,994 customer profiles (demographics, income, join date)
- **offers.csv** - 10 unique promotional offers (type, difficulty, reward, duration)
- **events.csv** - 306,648 transaction records (offer interactions, purchases)

Analysis focuses on 86,432 customer-offer interactions.

## Project Structure

```
Brewlytics_Chat/
├── Notebooks/
│   ├── 01_EDA.ipynb                      # Exploratory data analysis
│   ├── 02_Feature_Engg.ipynb             # Feature engineering & leakage prevention
│   ├── 03_Modeling.ipynb                 # Model training & selection
│   ├── 04_PCA.ipynb                      # Dimensionality reduction
│   ├── 05_SHAP.ipynb                     # Model explainability
│   ├── 06_Customer_Segmentation.ipynb    # K-means clustering
│   └── 07_Bias_Fairness_Analysis.ipynb   # Fairness auditing
├── Cafe_Rewards_Offers/
│   ├── customers.csv                     # Customer demographics
│   ├── offers.csv                        # Offer catalog
│   ├── events.csv                        # Transaction history
│   ├── models/                           # Trained models
│   ├── pca/                             # PCA transformers
│   ├── segmentation/                     # Cluster profiles
│   ├── fairness_analysis/               # Fairness metrics
│   └── processed/                        # Cleaned datasets
├── Data_Viz/                            # Visualization outputs
├── requirements.txt                      # Python dependencies
└── README.md                            # Project documentation
```

## Notebooks Workflow

### 1. Exploratory Data Analysis ([01_EDA.ipynb](Notebooks/01_EDA.ipynb))
- Load and merge customer, offer, and event data
- Analyze distributions, missing values, and data quality issues
- Identify relationships between features and offer completion
- Detect 26 raw features and data quality gaps

### 2. Feature Engineering ([02_Feature_Engg.ipynb](Notebooks/02_Feature_Engg.ipynb))
- **Remove data leakage:** Eliminate 8 features unavailable at prediction time
- **Encode categoricals:** One-hot encoding for offer type/gender, ordinal for age/income/tenure
- **Handle missing data:** Median imputation for 87 missing tenure values
- **Scale features:** StandardScaler for numerical features
- **Result:** Clean dataset with 24 features, 69,145 train / 17,287 test samples

### 3. Model Training & Selection ([03_Modeling.ipynb](Notebooks/03_Modeling.ipynb))
- Train and compare 4 classification algorithms
- **Best Model:** Random Forest (F1=0.8601, AUC=0.9277)
- Cross-validation confirms generalization (F1=0.8515 ± 0.0028)
- Top predictive features: offer type, tenure, income, offer duration
- Save production model with 84.5% test accuracy

### 4. Dimensionality Reduction ([04_PCA.ipynb](Notebooks/04_PCA.ipynb))
- Evaluate PCA with 6, 8, and 10 components
- **Recommended:** 8 components capturing 90% variance
- 67% fewer features with only 1.3% F1 performance drop
- Interpret principal components: tenure, demographics, offer characteristics

### 5. Model Explainability ([05_SHAP.ipynb](Notebooks/05_SHAP.ipynb))
- SHAP analysis reveals feature importance and directional impact
- **Top Features:** Offer type (49%), demographics (35%), tenure (19%)
- **Key Insights:**
  - Discount/BOGO offers drive completions
  - Shorter offers (<7 days) maximize completion
  - Higher income and longer tenure increase likelihood
  - Informational offers rarely complete (no reward)

### 6. Customer Segmentation ([06_Customer_Segmentation.ipynb](Notebooks/06_Customer_Segmentation.ipynb))
- K-means clustering identifies 3 distinct customer segments
- **Elite Performers (21%):** High income ($69K), long tenure (1.8y), 81.9% completion
- **Moderate Performers (66%):** Mid income ($64K), medium tenure (1.3y), 44.5% completion
- **At-Risk (13%):** Missing demographics, 14.1% completion - data quality issue
- Segment-specific marketing strategies recommended

### 7. Fairness Assessment ([07_Bias_Fairness_Analysis.ipynb](Notebooks/07_Bias_Fairness_Analysis.ipynb))
- Audit model fairness across gender, age, income, tenure groups
- Evaluate accuracy parity, false positive/negative rates, disparate impact
- **Finding:** Missing demographic data creates fairness issues (affects 12.8% of customers)
- Ensure equitable treatment across all customer demographics

## Key Features & Capabilities

### Machine Learning Pipeline
- Multi-model comparison (Logistic Regression, Decision Tree, Random Forest, XGBoost support)
- Hyperparameter tuning with GridSearchCV
- Cross-validation for robust performance estimation
- Production-ready models saved for deployment

### Analytical Capabilities
- Customer segmentation with K-means clustering
- Fairness auditing across demographic groups
- SHAP-based feature importance and prediction explanation
- Data quality assessment and gap identification

### Production Optimization
- All data leakage removed - features available at prediction time
- Optional PCA for 3x faster inference with minimal accuracy loss
- Saved transformers (scalers, encoders, PCA) for consistent preprocessing
- Comprehensive validation and documentation

## Installation & Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Brewlytics_Chat
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run notebooks in sequence:
```bash
jupyter notebook
# Navigate to Notebooks/ and run 01_EDA.ipynb through 07_Bias_Fairness_Analysis.ipynb
```

## Dependencies

Key libraries:
- **Data Processing:** pandas, numpy
- **Machine Learning:** scikit-learn, shap
- **Visualization:** matplotlib, seaborn, plotly
- **Statistical Analysis:** scipy, statsmodels

See [requirements.txt](requirements.txt) for complete list.

## Model Performance Summary

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **F1-Score** | 0.8601 | Balanced precision-recall (86% harmonic mean) |
| **Accuracy** | 84.5% | Correctly predicts 845 out of 1,000 offers |
| **Recall** | 89.1% | Catches 89% of customers who will complete |
| **Precision** | 83.2% | 83% of predicted completions are correct |
| **AUC-ROC** | 0.9277 | Excellent class separation ability |
| **CV F1** | 0.8515 ± 0.0028 | Consistent generalization across folds |

## Business Insights

### Top Predictive Factors
1. **Offer Type** - Discounts (+3.30) and BOGO (+1.85) drive completions; informational (-6.66) rarely complete
2. **Customer Tenure** - Long-term members (1.8y+) are 5x more likely to complete
3. **Income Level** - Higher-income customers ($65K+) show stronger engagement
4. **Offer Duration** - Offers under 7 days have highest completion rates

### Recommended Actions
- **Elite Segment:** Protect with VIP programs, leverage for referrals
- **Moderate Segment:** Optimize offer targeting - biggest growth opportunity
- **At-Risk Segment:** Fix data quality issues immediately to enable personalization
- **Offer Design:** Keep duration <7 days, focus on discount/BOGO over informational
- **Targeting:** Prioritize long-tenure, high-income customers for premium offers

## Future Enhancements

- Real-time prediction API for offer recommendation engine
- A/B testing framework for offer optimization
- Temporal features (seasonality, day-of-week effects)
- Ensemble models combining Random Forest with gradient boosting
- Advanced fairness constraints in model training
- Recommendation system for next-best-offer

## Contributing

This is an analytical project demonstrating end-to-end ML pipeline development. Contributions welcome for:
- Additional modeling techniques
- Enhanced feature engineering
- Advanced fairness metrics
- Production deployment guides

## License

Educational project for portfolio demonstration.

## Contact

For questions or collaboration opportunities, please open an issue in the repository.

---

**Last Updated:** January 2026
**Model Version:** Random Forest v1.0 (F1=0.8601)
