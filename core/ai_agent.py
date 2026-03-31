import os
from openai import OpenAI
from dotenv import load_dotenv

class BankingAIAgent:
    def __init__(self):
        """
        Initializes the Groq client.
        Checks Streamlit Secrets/System Environment first, then falls back to .env
        """
        # 1. Try to get key from System Environment (Streamlit Secrets)
        api_key = os.environ.get("GROQ_API_KEY")
        
        # 2. If not found, try loading from .env (Local development)
        if not api_key:
            load_dotenv()
            api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError("❌ Error: GROQ_API_KEY not found! Set it in Streamlit Secrets or a .env file.")

        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key
        )

    def get_marketing_insights(self, summary_stats, top_pos, top_neg):
        """
        Sends processed model data to Groq (Llama 3.3) to generate 
        strategic banking insights.
        """
        prompt = f"""
        You are an expert Banking Marketing Consultant. 
        
        Analysis of current campaign:
        - Sector Performance Breakdown: {summary_stats}
        - Top 5 Factors driving 'YES' (Successful Subscriptions): {top_pos}
        - Top 5 Factors driving 'NO' (Rejection/Failure): {top_neg}
        
        Please provide a detailed Strategic Report:
        1. STRATEGIC ANALYSIS: Explain WHY the top 5 positive factors are causing high customer engagement.
        2. RISK MITIGATION: Analyze the 5 negative/weak factors and suggest how the bank can improve them.
        3. SECTOR RECOVERY: Identify the worst-performing sectors and suggest specific policy changes or tailored offers to convert them.
        4. SALES TACTIC: Provide a tactical tip or 'script pivot' for the sales team to use during live calls.

        Keep the tone professional, data-driven, and actionable for a banking executive.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile", 
                messages=[
                    {{"role": "system", "content": "You are a professional banking strategist and marketing expert."}},
                    {{"role": "user", "content": prompt}},
                ],
                temperature=0.7,
                max_tokens=1024
            )
            return response.choices[0].message.content
            
        except Exception as e:
            return f"🤖 AI Agent Error: {str(e)}"