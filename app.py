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
    # Simple line wrapping logic for the PDF
    for line in report_text.split('\n'):
        # If line is too long, it might bleed off; for a pro version, use Platypus
        textobject.textLine(line)
    c.drawText(textobject)
    
    c.showPage()
    c.save()
    buf.seek(0)
    return buf

# --- 4. AUTHENTICATION & RECOVERY ---
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
            st.write("Please contact IT support to reset your credentials.")
            st.info("Support Email: admin@bankai.com")
            if st.button("Back to Login"):
                st.session_state["show_recovery"] = False
                st.rerun()
    return False

# --- 5. MAIN APP ---
if check_password():
    # Sidebar Info
    with st.sidebar:
        if os.path.exists("logo.png"):
            st.image("logo.png", width=100)
        st.write(f"👤 **{st.session_state['username']}**")
        st.divider()
        st.subheader("Contact & Support")
        st.write("📧 support@custoview.in")
        st.write("📞 +91-0712-751180")
        st.write("📍 Wellesley Road, Shivajinagar, Pune: 411005, IN")
        
        if st.button("Logout"):
            st.session_state["password_correct"] = False
            st.session_state["username"] = "Guest"
            st.rerun()

    st.title("🏦 BankAI Policy Dashboard")

    uploaded_file = st.file_uploader("Upload test.csv", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file, sep=';')
        
        if st.button("Execute Full AI Analysis"):
            log_action(st.session_state["username"], "Executed Full Analysis")
            try:
                # 1. PREPROCESSING (Uses DataProcessor to drop 'month'/'day' and 'y')
                processor = DataProcessor()
                X_encoded = processor.clean_and_encode(df, is_train=False)
                
                # 2. LOAD MODEL & PREDICT
                if not os.path.exists('models/stack_model.pkl'):
                    st.error("Model file not found! Please run main.py first.")
                    st.stop()
                    
                model = joblib.load('models/stack_model.pkl')
                
                # Run Predictions
                preds = model.predict(X_encoded)
                df['Prediction'] = ['Likely YES' if p == 1 else 'Unlikely NO' for p in preds]

                # 3. EXTRACT IMPORTANCE
                xgb_model = model.named_estimators_['xgb']
                feat_importance_df = pd.DataFrame({
                    'Feature': X_encoded.columns,
                    'Importance': xgb_model.feature_importances_
                }).sort_values(by='Importance', ascending=False)

                top_5_pos = feat_importance_df.head(5)['Feature'].tolist()
                top_5_neg = feat_importance_df.tail(5)['Feature'].tolist()

                # 4. VISUALS
                st.header("📊 Data Breakdown")
                c1, c2 = st.columns(2)
                with c1:
                    fig_imp = px.bar(feat_importance_df.head(10), x='Importance', y='Feature', 
                                   orientation='h', title="Key Decision Factors",
                                   color_discrete_sequence=['#1f77b4'])
                    st.plotly_chart(fig_imp, use_container_width=True)
                with c2:
                    fig_job = px.histogram(df, x="job", color="Prediction", barmode="group",
                                         title="Customer Response by Sector")
                    st.plotly_chart(fig_job, use_container_width=True)

                # 5. AI STRATEGY & PDF
                st.divider()
                st.header("🤖 AI Strategic Insights")
                
                with st.spinner("AI Agent is generating report..."):
                    # Calculate sectors with high "No" counts
                    low_perf = df[df['Prediction'] == 'Unlikely NO']['job'].value_counts().head(3).to_dict()
                    
                    agent = BankingAIAgent()
                    ai_report = agent.get_marketing_insights(str(low_perf), top_5_pos, top_5_neg)
                    
                    st.markdown(ai_report)
                    st.session_state['current_report'] = ai_report

                if 'current_report' in st.session_state:
                    pdf_fp = create_pdf(st.session_state['current_report'])
                    st.download_button(
                        label="📄 Download AI Report as PDF",
                        data=pdf_fp,
                        file_name="BankAI_Strategy_Report.pdf",
                        mime="application/pdf"
                    )
                
                st.divider()
                st.subheader("Targeted Leads (Probable Subscribers)")
                st.dataframe(df[df['Prediction'] == 'Likely YES'].head(20))

            except Exception as e:
                st.error(f"Error during analysis: {e}")