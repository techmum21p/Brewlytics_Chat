# Brewlytics Offer Completion Prediction App

A Streamlit web application that predicts which customers will complete offers using a trained Random Forest model.

## Features

- 📤 Upload CSV files with customer and offer data
- 🤖 Generate predictions using the trained Random Forest model
- 📊 View prediction statistics and probability distributions
- 📥 Download results with predictions and probabilities
- 🐳 Docker containerized for easy deployment

## Quick Start

### Option 1: Run Locally with Python

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the app:**
   ```bash
   streamlit run app.py
   ```

3. **Open in browser:**
   Navigate to `http://localhost:8501`

### Option 2: Run with Docker

1. **Build the Docker image:**
   ```bash
   docker build -t brewlytics-app .
   ```

2. **Run the container:**
   ```bash
   docker run -p 8501:8501 -v $(pwd)/../Cafe_Rewards_Offers:/app/Cafe_Rewards_Offers brewlytics-app
   ```
   
   The `-v` flag mounts the Cafe_Rewards_Offers directory so the app can access the model and scaler files.

3. **Open in browser:**
   Navigate to `http://localhost:8501`

## Required CSV Columns

Your CSV file should contain the following columns:

### Offer Details
- `received_time` - Time the offer was received
- `difficulty` - Offer difficulty level
- `duration` - Offer duration in days
- `offer_type` - Type of offer (bogo, discount, informational)

### Marketing Channels
- `in_email` - Offer sent via email (0/1)
- `in_mobile` - Offer sent via mobile (0/1)
- `in_social` - Offer sent via social media (0/1)
- `in_web` - Offer sent via web (0/1)
- `offer_received` - Offer received flag (0/1)

### Customer Demographics
- `age` - Customer age
- `income` - Customer income
- `gender` - Customer gender (F, M, O, Missing)
- `age_group` - Age group (18-30, 31-45, 46-60, 61-75, 76+)
- `income_bracket` - Income bracket (Missing, Low, Medium, High, Very High)

### Membership Information
- `membership_year` - Year customer became member
- `membership_duration_days` - Days since membership started
- `membership_month` - Month customer became member
- `tenure_group` - Customer tenure group (0-6 months, 6-12 months, 1-2 years, 2+ years)

### Flags
- `is_demographics_missing` - Flag for missing demographics (0/1)

## Model Information

- **Model Type:** Random Forest Classifier
- **Accuracy:** 84.54%
- **F1-Score:** 0.8601
- **AUC-ROC:** 0.9277
- **Features:** 24 features

## Output

The app generates predictions with the following columns:
- `prediction` - Binary prediction (0: Will Not Complete, 1: Will Complete)
- `prediction_label` - Human-readable prediction label
- `completion_probability` - Probability of completing the offer (0-1)
- `non_completion_probability` - Probability of not completing the offer (0-1)

## Docker Deployment Notes

The app requires access to the trained model and preprocessing artifacts located in the `../Cafe_Rewards_Offers/` directory. When running with Docker, you must mount this directory as a volume.

### Example Docker Compose

```yaml
version: '3.8'
services:
  brewlytics:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ../Cafe_Rewards_Offers:/app/Cafe_Rewards_Offers
    restart: unless-stopped
```

Run with:
```bash
docker-compose up
```

## Streamlit vs FastAPI

You asked about containerizing Streamlit vs FastAPI. Both can be containerized with Docker:

**Why I chose Streamlit:**
- Simpler for this use case (file upload, display, download)
- Built-in UI components (no need for separate frontend)
- Excellent for data visualization
- Lower development time and complexity
- Perfect for interactive data science apps

**When to choose FastAPI:**
- Building a REST API for integration with other systems
- Need more control over frontend UI (React, Vue, etc.)
- Building microservices architecture
- Need WebSocket support
- Custom authentication/authorization beyond basic

For this prediction app where users upload a file and see results, Streamlit is the ideal choice.

## Troubleshooting

**Error: "Model file not found"**
- Ensure the `Cafe_Rewards_Offers/models/random_forest.pkl` file exists
- When using Docker, verify the volume mount is correct

**Error: "Error during prediction"**
- Check that your CSV contains all required columns
- Verify data types match the training data format

**App is slow:**
- The Random Forest model is ~200MB, first prediction may take a few seconds
- Subsequent predictions are faster due to caching
