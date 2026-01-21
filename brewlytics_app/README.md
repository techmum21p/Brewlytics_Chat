# Brewlytics Offer Completion Prediction App

A Streamlit web application that predicts which customers will complete offers using a trained Random Forest model. This project demonstrates local model deployment with MLOps best practices.

## Features

- 📤 Upload CSV files with customer and offer data
- 🤖 Generate predictions using the trained Random Forest model
- 📊 View prediction statistics and probability distributions
- 📥 Download results with predictions and probabilities
- 🎯 Dynamic Marketing Strategy Generator powered by user-selected LLMs
  - Support for Google Gemini, OpenAI, and Anthropic Claude
  - Personalized marketing strategies based on customer segments
  - Integration with customer clustering and segmentation data
- 🐳 Docker containerized for easy deployment

---

## Local Deployment Guide

This section provides step-by-step instructions for deploying the trained Random Forest model locally via Streamlit.

### Prerequisites

Before starting, ensure you have the following installed:

| Requirement | Version | Check Command |
|-------------|---------|---------------|
| Python | 3.10+ | `python --version` |
| pip | Latest | `pip --version` |
| Docker (optional) | 20.10+ | `docker --version` |
| Docker Compose (optional) | 2.0+ | `docker-compose --version` |

### Project Structure

```
brewlytics_app/
├── app.py                  # Main Streamlit application
├── Dockerfile              # Docker image configuration
├── docker-compose.yml      # Docker Compose orchestration
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── ../Cafe_Rewards_Offers/ # Model artifacts directory
    ├── models/
    │   └── random_forest.pkl       # Trained model (~200MB)
    ├── processed/
    │   ├── scaler.pkl              # Feature scaler
    │   └── feature_names.pkl       # Feature list
    └── segmentation/
        ├── customers_with_clusters.csv
        └── segment_profiles.csv
```

---

### Option 1: Local Deployment with Python (Recommended for Development)

**Step 1: Clone and navigate to the project**
```bash
cd brewlytics_app
```

**Step 2: Create a virtual environment (recommended)**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
.\venv\Scripts\activate
```

**Step 3: Install dependencies**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Step 4: Verify model artifacts exist**
```bash
# Check that required files are present
ls ../Cafe_Rewards_Offers/models/random_forest.pkl
ls ../Cafe_Rewards_Offers/processed/scaler.pkl
ls ../Cafe_Rewards_Offers/processed/feature_names.pkl
```

**Step 5: Run the application**
```bash
streamlit run app.py
```

**Step 6: Access the application**
- Open your browser and navigate to: `http://localhost:8501`
- The app should display the Brewlytics prediction interface

---

### Option 2: Local Deployment with Docker (Recommended for Production)

Docker ensures a reproducible environment across different machines.

**Step 1: Navigate to the project directory**
```bash
cd brewlytics_app
```

**Step 2: Build the Docker image**
```bash
docker build -t brewlytics-app .
```

**Step 3: Run the container**
```bash
docker run -p 8501:8501 -v $(pwd)/../Cafe_Rewards_Offers:/app/Cafe_Rewards_Offers brewlytics-app
```

The `-v` flag mounts the model artifacts directory into the container.

**Step 4: Access the application**
- Open your browser: `http://localhost:8501`

---

### Option 3: Docker Compose (Easiest Setup)

Docker Compose simplifies multi-container setups and configuration.

**Step 1: Navigate to the project directory**
```bash
cd brewlytics_app
```

**Step 2: Start the application**
```bash
docker-compose up --build
```

**Step 3: Access the application**
- Open your browser: `http://localhost:8501`

**Step 4: Stop the application**
```bash
docker-compose down
```

---

## MLOps Practices Implemented

### 1. Reproducible Environments

| Component | Implementation |
|-----------|----------------|
| **requirements.txt** | Pins all Python dependencies for consistent installations |
| **Dockerfile** | Ensures identical runtime environment across machines |
| **docker-compose.yml** | Orchestrates services with consistent configuration |

### 2. Configuration-Driven Design

- Model paths are configurable via environment variables
- LLM provider selection is runtime-configurable through the UI
- Docker Compose environment variables allow easy customization

### 3. Model Versioning & Artifacts

| Artifact | Path | Description |
|----------|------|-------------|
| Trained Model | `Cafe_Rewards_Offers/models/random_forest.pkl` | Random Forest classifier |
| Feature Scaler | `Cafe_Rewards_Offers/processed/scaler.pkl` | StandardScaler for feature normalization |
| Feature Names | `Cafe_Rewards_Offers/processed/feature_names.pkl` | List of 24 features used by model |
| Segment Data | `Cafe_Rewards_Offers/segmentation/` | Customer clustering data |

### 4. Rollback Plan

To rollback to a previous model version:

1. **Backup current model:**
   ```bash
   cp Cafe_Rewards_Offers/models/random_forest.pkl Cafe_Rewards_Offers/models/random_forest_backup.pkl
   ```

2. **Restore previous version:**
   ```bash
   cp Cafe_Rewards_Offers/models/random_forest_v1.pkl Cafe_Rewards_Offers/models/random_forest.pkl
   ```

3. **Restart the application:**
   ```bash
   # If using Docker Compose
   docker-compose restart

   # If using Python directly
   # Stop the running app (Ctrl+C) and restart
   streamlit run app.py
   ```

### 5. Health Monitoring

The Docker deployment includes a health check endpoint:
- **Endpoint:** `http://localhost:8501/_stcore/health`
- **Check command:** `curl --fail http://localhost:8501/_stcore/health`

Monitor application status:
```bash
# Check container health status
docker ps

# View application logs
docker-compose logs -f brewlytics
```

---

## Deployment Verification Checklist

After deployment, verify the application is working correctly:

- [ ] Application loads at `http://localhost:8501`
- [ ] Model artifacts load without errors (no "Model file not found" message)
- [ ] Can upload a sample CSV file
- [ ] Predictions generate successfully
- [ ] Download button works for results
- [ ] LLM integration works (if API key configured)

---

## Troubleshooting

### Common Issues

**Issue: "Model file not found"**
```bash
# Verify model file exists
ls -la ../Cafe_Rewards_Offers/models/random_forest.pkl

# If using Docker, check volume mount
docker inspect <container_id> | grep Mounts -A 20
```

**Issue: Port 8501 already in use**
```bash
# Find process using port
lsof -i :8501

# Kill the process or use a different port
streamlit run app.py --server.port=8502
```

**Issue: Docker volume mount not working (Windows)**
```bash
# Use absolute path on Windows
docker run -p 8501:8501 -v C:/path/to/Cafe_Rewards_Offers:/app/Cafe_Rewards_Offers brewlytics-app
```

**Issue: Dependencies installation fails**
```bash
# Clear pip cache and retry
pip cache purge
pip install -r requirements.txt --no-cache-dir
```

---

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

### Dynamic Marketing Strategy Generator

After generating predictions, you can use the Dynamic Marketing Strategy Generator to create personalized marketing strategies:

1. **Configure LLM in Sidebar**:
   - Select your preferred LLM provider (Google Gemini, OpenAI, or Anthropic Claude)
   - Enter your API key securely (password field, not stored)
   - Choose the specific model to use

2. **Generate Strategies**:
   - Select the number of customers to generate strategies for
   - Choose to target: "Most Likely to Complete", "Least Likely to Complete", or "All Customers"
   - Adjust batch size for API rate limits (smaller is safer)
   - Click "Generate Marketing Strategies"

3. **Strategy Generation Process**:
   - For each customer, the system:
     - Matches them to a customer cluster based on demographics
     - Retrieves segment profile with historical behavior patterns
     - Sends customer data and segment context to the selected LLM
     - Generates a personalized marketing strategy to ensure offer completion

4. **Output**:
   - View strategies in a table with customer details and recommendations
   - Download results as CSV with `recommended_strategy` column
   - Includes segment information and completion probability

**Data Sources**:
- `Cafe_Rewards_Offers/segmentation/customers_with_clusters.csv` - Maps customers to clusters
- `Cafe_Rewards_Offers/segmentation/segment_profiles.csv` - Segment descriptions and strategies

**Supported LLMs**:
- Google Gemini 2.5 Flash/Pro, 1.5 Flash/Pro
- OpenAI GPT-4o, GPT-4 Turbo, GPT-3.5 Turbo
- Anthropic Claude 3 Haiku/Sonnet/Opus

## Framework Choice: Streamlit vs FastAPI

**Why Streamlit was chosen for this project:**
- Simpler for this use case (file upload, display, download)
- Built-in UI components (no need for separate frontend)
- Excellent for data visualization
- Lower development time and complexity
- Perfect for interactive data science apps

**When to choose FastAPI instead:**
- Building a REST API for integration with other systems
- Need more control over frontend UI (React, Vue, etc.)
- Building microservices architecture
- Need WebSocket support
- Custom authentication/authorization beyond basic
