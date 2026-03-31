import os
from openai import OpenAI
from dotenv import load_dotenv

class BankingAIAgent:
    def __init__(self):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            load_dotenv()
            api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError("❌ Error: GROQ_API_KEY not found!")

        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key
        )

    def get_marketing_insights(self, summary_stats, top_factors):
        """
        Generates insights by analyzing the most influential features.
        """
        prompt = f"""
        You are an expert Banking Marketing Consultant. 
        
        Analysis of current campaign data:
        - Poorly Performing Sectors: {str(summary_stats)}
        - Top 10 Most Influential Features: {str(top_factors)}
        
        CRITICAL ANALYSIS REQUIREMENT:
        Do not assume all influential features are positive. For example, 'housing' or 'loan' usually indicate existing debt, which often DECREASES a customer's likelihood to invest in a Term Deposit.
        
        Please provide a detailed Strategic Report:
        1. FEATURE ANALYSIS: Identify which of the top 10 features are likely 'Triggers' (driving 'Yes') and which are 'Barriers' (driving 'No', like high debt or low balance).
        2. RISK MITIGATION: Suggest how to approach customers with 'Barriers' (e.g., those with housing loans).
        3. SECTOR RECOVERY: Provide a recovery plan for the worst-performing sectors.
        4. SALES TACTIC: Provide a tactical 'script pivot' for the sales team.

        Tone: Professional, data-driven, and executive.
        """
        
        try:
            messages = [
                {"role": "system", "content": "You are a professional banking strategist and expert data analyst."},
                {"role": "user", "content": prompt}
            ]
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile", 
                messages=messages,
                temperature=0.4, # Lower temperature for higher factual accuracy
                max_tokens=1200
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"🤖 AI Agent Error: {str(e)}"