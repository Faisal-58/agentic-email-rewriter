from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from crewai import LLM, Agent, Task, Crew
from crewai.tools import BaseTool
from transformers import pipeline
from pydantic import PrivateAttr

#Sentiment Analysis Tool using Hugging Face
class SentimentAnalysisTool(BaseTool):
    name: str = "Email Sentiment Analyzer"
    description: str = "Detects the overall sentiment of the email using a pre-trained model"

    _sentiment_pipeline = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        self._sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model_name,
            tokenizer=model_name,
            device=-1  
        )

    def _run(self, email: str) -> str:
        result = self._sentiment_pipeline(email)[0]
        label = result["label"]
        score = result["score"]
        return f"{label} ({score:.2f})"

# Streamlit UI Setup
st.set_page_config(page_title="Email Rewriter", page_icon="📧", layout="centered")
st.markdown("""
    <style>
    /* Change entire app background */
    .stApp {
        background-color: #f7f9fc !important;
        font-family: 'Segoe UI', sans-serif;
    }

    /* White container in the middle */
    .main-container {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0px 4px 20px rgba(0,0,0,0.05);
        max-width: 800px;
        margin: auto;
    }

    /* Section headings */
    .section-title {
        font-size: 1.4rem;
        font-weight: 600;
        color: #333;
        margin-bottom: 10px;
    }

    /* Sentiment badges */
    .sentiment-badge {
        padding: 6px 12px;
        border-radius: 12px;
        font-weight: 500;
        display: inline-block;
    }
    .positive { background: #d4edda; color: #155724; }
    .negative { background: #f8d7da; color: #721c24; }
    .neutral  { background: #fff3cd; color: #856404; }
    </style>
""", unsafe_allow_html=True)

st.title("📧 Professional Email Rewriter with Sentiment Analysis")
st.markdown("Turn your raw emails into **professional, polished messages** and see the **sentiment instantly**.")
email_input = st.text_area("Paste your email here:", height=180)

tone_options = ["Formal", "Friendly", "Concise", "Empathetic"]
selected_tone = st.selectbox("Choose the tone for the rewrite:", tone_options)

if st.button("Rewrite Email"):
    if not email_input.strip():
        st.warning("Please paste an email first.")
    else:
        # Sentiment detection
        sentiment_tool = SentimentAnalysisTool()
        detected_sentiment = sentiment_tool._run(email_input)
        sentiment_label = detected_sentiment.split()[0].lower()

        # LLM for rewriting
        llm = LLM(model="gemini/gemini-2.0-flash", temperature=0.1)

        email_assistant = Agent(
            role="Professional Email Assistant",
            goal=f"Rewrite emails into professional, clear messages with a {selected_tone.lower()} tone.",
            backstory="An expert in transforming informal or unclear emails into polished communication.",
            llm=llm
        )

        task = Task(
            description=f"""
            Rewrite the following email into a {selected_tone.lower()} professional, polished version while keeping the meaning:

            Email:
            '''{email_input}'''
            """,
            agent=email_assistant,
            expected_output=f"A {selected_tone.lower()} professional rewritten version of the email."
        )

        crew = Crew(agents=[email_assistant], tasks=[task])
        result = crew.kickoff()

        if result.tasks_output:
            rewritten_email = result.tasks_output[0].raw
        else:
            rewritten_email = str(result)

        # Display results
        
        st.subheader(f" Professional Rewrite ({selected_tone} tone)")
        
        st.write(rewritten_email)
        st.download_button(
               label="📥 Download Rewritten Email",
               data=rewritten_email,
               file_name="rewritten_email.txt",
               mime="text/plain")
        
        sentiment_class = "neutral"
        if "positive" in sentiment_label:
            sentiment_class = "positive"
        elif "negative" in sentiment_label:
            sentiment_class = "negative"

        st.markdown("<div class='section-title'>📊 Sentiment</div>", unsafe_allow_html=True)
        st.markdown(f"<span class='sentiment-badge {sentiment_class}'>{detected_sentiment}</span>", unsafe_allow_html=True)
