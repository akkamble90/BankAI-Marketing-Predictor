# 🏦 BankAI: Predictive Marketing & Strategic AI Agent

An End-to-End Machine Learning pipeline that predicts bank subscription likelihood using a **Stacking Ensemble (XGBoost + Random Forest)** and provides strategic business insights via a **Groq-powered AI Agent**. BankAI is a comprehensive machine learning and Generative AI solution designed to optimize bank telemarketing campaigns. By combining a high-performance Stacking Ensemble with a Llama 3.3 Strategic Agent, this project transforms raw customer data into prioritized lead lists and executive-level business intelligence.

Technical Architecture & Data Science Workflow

1. Data Preprocessing & Feature Engineering

To achieve high model stability, the following steps were taken:

* Dimensionality Reduction: Removed seasonal noise features such as month and day to ensure the model focuses on customer financial health and demographics rather than call timing.
* Target Encoding: Leveraged category_encoders to transform high-cardinality categorical features such as job and education into numerical values based on their statistical relationship with the target variable.
* Class Balancing (SMOTE): Addressed the severe class imbalance where "No" subscriptions heavily outweighed "Yes" using Synthetic Minority Over-sampling Technique. This prevents the model from being biased toward the majority class.

2. Model Training & Optimization

* Hyperparameter Tuning (Optuna): Utilized the Optuna framework for Bayesian optimization. Tuned learning_rate, max_depth, and n_estimators for the XGBoost component to minimize log-loss.
* High Accuracy Strategy: Achieved an impressive balance between training and test accuracy of approximately 90% or more. By using 5-fold cross-validation during tuning, the generalization gap stayed within a professional margin of less than 15% to prevent overfitting.

3. The Stacking Ensemble

Instead of using a single model, a Stacking Classifier was implemented to maximize predictive power.

* Base Models: XGBoost and Random Forest
* Meta-Learner: Logistic Regression

The meta-learner learns how to best combine the predictions of the base models, resulting in a robust system that captures both linear and non-linear patterns in the data.

AI Strategic Agent (Groq + Llama 3.3)

After the model identifies who will subscribe, the Banking AI Agent explains why and how to close the deal.

Strategic Insights Discovered:

* Debt Impact: The model identified housing loans as the primary negative driver. Customers with significant existing debt are statistically 25% less likely to commit to new term deposits.
* The Loyalty Loop: Previous successful interactions (poutcome) were the strongest predictors of future success, suggesting a quality over quantity approach for the sales team.

Rectifying AI Hallucinations:

To ensure factual accuracy in a sensitive banking environment:

* Context Injection: A strict prompt engineering template injects real-time summary statistics and feature importance rankings directly into the LLM.
* Persona Grounding: The agent is locked into a Professional Banking Strategist role and instructed to only provide insights derived from the uploaded dataset.
* Deterministic Controls: Temperature is set to 0.7 to allow for tactical creativity in sales scripts while maintaining factual consistency in the data report.

Installation & Deployment

1. Clone the Repository

git clone https://github.com/akkamble90/BankAI-Marketing-Predictor.git

2. Install Dependencies

pip install -r requirements.txt

3. Setup Environment

Create a .env file or add the following to Streamlit Secrets:

GROQ_API_KEY = "your_key_here"

4. Run the App

streamlit run app.py

Future Roadmap

* SHAP Integration: Add individual-level local explanations for every customer prediction.
* Real-time Retraining: Allow users to trigger a new Optuna study directly from the UI.
* LFS Implementation: Move large model binaries to Git Large File Storage for cleaner version control.

Dataset Credits

This project utilizes the UCI Bank Marketing Dataset, which contains results of phone-based marketing campaigns by a Portuguese banking institution.
