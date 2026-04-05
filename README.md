# Cognitive Lifestyle Analysis

This project explores how lifestyle and behavioral factors are associated with cognitive performance using data analysis and machine learning techniques.

The goal is not only to predict a cognitive score, but also to understand which variables are most relevant and how a structured machine learning pipeline can support interpretable insights.

---

## Dataset

The dataset includes demographic, behavioral, and cognitive-related variables such as:

- Age
- Gender
- Sleep duration
- Stress level
- Diet type
- Daily screen time
- Exercise frequency
- Caffeine intake
- Reaction time
- Memory test score
- Cognitive score
- AI-predicted score

### Target variable
- `Cognitive_Score`

### Excluded variables
- `AI_Predicted_Score` → used only as benchmark to avoid data leakage
- `User_ID` → not informative for prediction

---

## Project Goals

- Explore relationships between lifestyle factors and cognitive performance
- Build a complete and clean machine learning pipeline
- Compare model performance with:
  - a baseline (Dummy Regressor)
  - an external benchmark (AI predicted score)
- Identify the most influential features

---

## Tech Stack

- Python
- pandas
- scikit-learn
- matplotlib
- Jupyter Notebook

---

## Project Structure
cognitive-lifestyle-analysis/
│
├── data/
├── notebooks/
│ └── exploration.ipynb
├── src/
├── main.py
├── requirements.txt
├── README.md
└── .gitignore


---

## Workflow

1. Exploratory Data Analysis (EDA)
2. Correlation analysis
3. Feature selection and leakage prevention
4. Preprocessing pipeline:
   - numerical scaling (MinMaxScaler)
   - categorical encoding (OneHotEncoder)
5. Model training:
   - Random Forest Regressor
6. Model evaluation:
   - Cross-validation
   - Test set evaluation
7. Model comparison:
   - Dummy Regressor (baseline)
   - AI benchmark

---

## Key Insights

- Reaction time is the most influential feature in predicting cognitive performance
- Memory test score shows a strong positive relationship with cognitive score
- Stress level has a moderate negative impact
- Exercise frequency emerges as an important feature in the model, suggesting non-linear relationships not captured by simple correlation analysis

---

## Results

| Model | MAE | RMSE | R² |
|------|-----:|-----:|----:|
| Dummy | 19.1099 | 22.9569 | -0.0000 |
| Random Forest | 1.9937 | 2.5271 | 0.9879 |
| AI Benchmark | 2.4301 | 2.8384 | 0.9847 |

---

## Model Interpretation

The Random Forest model significantly outperforms the baseline, confirming that it is learning meaningful patterns from the data.

It also slightly outperforms the AI benchmark, showing that a relatively simple model can effectively capture the underlying relationships.

Feature importance analysis partially confirms the initial hypothesis:
- Reaction time and memory test score are dominant predictors
- Exercise frequency appears important despite low linear correlation, highlighting the model's ability to capture non-linear effects

---

## Limitations

- The dataset appears relatively easy due to highly predictive features
- Results may not generalize to real-world cognitive modeling scenarios
- No causal inference is performed

---

## How to Run

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt

3. Run the project:

python main.py

Future Improvements
Compare additional models (e.g., Linear Regression)
Add model explainability (e.g., SHAP values)
Build an interactive dashboard (Streamlit)
Test on more complex and realistic datasets