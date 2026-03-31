import streamlit as st
import pandas as pd
import joblib
import os
import logging
import plotly.express as px
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from core.ai_agent import BankingAIAgent 
from core.processor import DataProcessor 

# --- 0. PAGE CONFIG ---
st.set_page_config(page_title="BankAI - Smart Marketing", layout="wide")

# --- 1. INITIALIZATION ---
if "password_correct" not in st.session_state:
    st.session_state["password_correct"] = False
if "username" not in st.session_state:
    st.session_state["username"] = "Guest"
if "show_recovery" not in st.session_state:
    st.session_state["show_recovery"] = False
if "current_report" not in st.session_state:
    st.session_state["current_report"] = None
# This persists the data after predictions so buttons don't cause a crash
if "predicted_df" not in st.session_state:
    st.session_state["predicted_df"] = None

# --- 2. LOGGING ---
os.makedirs('logs', exist_ok=True)
logging.basicConfig(filename='logs/app_activity.log', level=logging.INFO, format='%(asctime)s - %(message)s')

def log_action(user, action):
    logging.info(f"User: {user} | Action: {action}")

# --- 3. PDF GENERATOR FUNCTION ---
def create_pdf(report_text):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 750, "BankAI Strategic Marketing Report")
    
    c.setFont("Helvetica", 10)
    textobject = c.beginText(100, 720)
    for line in report_text.split('\n'):
        textobject.textLine(line[:100]) 
    c.drawText(textobject)
    
    c.showPage()
    c.save()
    buf.seek(0)
    return buf

# --- 4. AUTHENTICATION ---
def check_password():
    if st.session_state["password_correct"]:
        return True

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if os.path.exists("logo.png"):
            st.image("logo.png", width=150)
        st.title("BankAI Login")
        
        if not st.session_state["show_recovery"]:
            user_input = st.text_input("Username")
            pass_input = st.text_input("Password", type="password")
            
            if st.button("Login"):
                if user_input == "admin" and pass_input == "bank123":
                    st.session_state["password_correct"] = True
                    st.session_state["username"] = user_input
                    log_action(user_input, "Logged In")
                    st.rerun()
                else:
                    st.error("Invalid credentials")
            
            if st.button("Forgot Username or Password?"):
                st.session_state["show_recovery"] = True
                st.rerun()
        else:
            st.subheader("Credential Recovery")
            st.info("Support Email: admin@bankai.com")
            if st.button("Back to Login"):
                st.session_state["show_recovery"] = False
                st.rerun()
    return False

# --- 5. MAIN APP ---
if check_password():
    with st.sidebar:
        if os.path.exists("logo.png"):
            st.image("logo.png", width=100)
        st.write(f"👤 **{st.session_state['username']}**")
        st.divider()
        st.subheader("Contact & Support")
        st.write("📧 support@castoview.in")
        st.write("📍 Wellesley Road, Shivajinagar, Pune: 411005, IN")
        st.write("📞 +91-0712-751180")
        
        if st.button("Logout"):
            st.session_state["password_correct"] = False
            st.session_state["predicted_df"] = None
            st.session_state["current_report"] = None
            st.rerun()

    st.title("🏦 BankAI Policy Dashboard")
    st.write("Upload campaign data to predict customer subscriptions and generate AI strategies.")

    uploaded_file = st.file_uploader("Upload campaign CSV (Semicolon separated)", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file, sep=';')
        
        if st.button("Execute Full AI Analysis"):
            log_action(st.session_state["username"], "Executed Full Analysis")
            try:
                # 1. PREPROCESSING
                processor = DataProcessor()
                X_encoded = processor.clean_and_encode(df, is_train=False)
                
                # 2. LOAD MODEL
                if not os.path.exists('models/stack_model.pkl'):
                    st.error("❌ Model not found! Please push models/stack_model.pkl to GitHub.")
                    st.stop()
                    
                model = joblib.load('models/stack_model.pkl')
                
                # 3. PREDICTIONS
                preds = model.predict(X_encoded)
                df['Prediction'] = ['Likely YES' if p == 1 else 'Unlikely NO' for p in preds]
                
                # Store in session state so it survives the download button rerun
                st.session_state["predicted_df"] = df

                # 4. EXTRACT TOP 10 INFLUENTIAL FEATURES
                xgb_model = model.named_estimators_['xgb']
                feat_importance_df = pd.DataFrame({
                    'Feature': X_encoded.columns,
                    'Importance': xgb_model.feature_importances_
                }).sort_values(by='Importance', ascending=False)

                top_10_factors = feat_importance_df.head(10)['Feature'].tolist()

                # 5. VISUALS
                st.header("📊 Data Breakdown")
                c1, c2 = st.columns(2)
                with c1:
                    fig_imp = px.bar(feat_importance_df.head(10), x='Importance', y='Feature', 
                                   orientation='h', title="Key Decision Factors (Influence)",
                                   color_discrete_sequence=['#1f77b4'])
                    st.plotly_chart(fig_imp, use_container_width=True)
                with c2:
                    fig_job = px.histogram(df, x="job", color="Prediction", barmode="group",
                                         title="Customer Response by Sector")
                    st.plotly_chart(fig_job, use_container_width=True)

                # 6. AI STRATEGY
                st.divider()
                st.header("🤖 AI Strategic Insights")
                
                with st.spinner("AI Agent is generating report based on your model's findings..."):
                    low_perf_dict = df[df['Prediction'] == 'Unlikely NO']['job'].value_counts().head(3).to_dict()
                    agent = BankingAIAgent()
                    ai_report = agent.get_marketing_insights(low_perf_dict, top_10_factors)
                    st.markdown(ai_report)
                    st.session_state['current_report'] = ai_report

            except Exception as e:
                st.error(f"⚠️ Error during analysis: {str(e)}")

        # --- DOWNLOAD & LEADS SECTION (Using Session State) ---
        if st.session_state["predicted_df"] is not None:
            # We work with the saved dataframe to avoid KeyError
            res_df = st.session_state["predicted_df"]
            
            st.divider()
            col_a, col_b = st.columns([1, 1])
            
            with col_a:
                if st.session_state['current_report']:
                    pdf_fp = create_pdf(st.session_state['current_report'])
                    st.download_button(
                        label="📄 Download AI Report as PDF",
                        data=pdf_fp,
                        file_name="BankAI_Strategy_Report.pdf",
                        mime="application/pdf"
                    )
            
            with col_b:
                csv = res_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Full Predictions CSV",
                    data=csv,
                    file_name="bank_predictions.csv",
                    mime="text/csv"
                )

            st.subheader("Targeted Leads (Top 20 Probable Subscribers)")
            # Safely filter the persisted dataframe
            st.dataframe(res_df[res_df['Prediction'] == 'Likely YES'].head(20))