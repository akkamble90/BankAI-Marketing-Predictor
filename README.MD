# 🏦 BankAI: Predictive Marketing & Strategic AI Agent

An End-to-End Machine Learning pipeline that predicts bank subscription likelihood using a **Stacking Ensemble (XGBoost + Random Forest)** and provides strategic business insights via a **Groq-powered AI Agent**.

## 🚀 Features
- **Stacking Classifier:** High-precision prediction of customer behavior.
- **AI Strategy Agent:** Uses Llama 3.3 (via Groq) to provide policy recommendations.
- **Interactive Dashboard:** Streamlit UI with Plotly visualizations.
- **PDF Reporting:** Automated generation of AI Strategic reports.

## 🛠️ Installation
1. Clone the repo.
2. Install dependencies: `pip install -r requirements.txt`
3. Add your `GROQ_API_KEY` to a `.env` file.
4. Run training: `python main.py`
5. Launch App: `streamlit run app.py`