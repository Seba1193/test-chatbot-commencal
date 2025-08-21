import streamlit as st
import os
from openai import OpenAI

openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    st.error("Missing OPENAI_API_KEY. Export it in your terminal (e.g., export OPENAI_API_KEY=\"sk-...\") and rerun.")
    st.stop()

client = OpenAI(api_key=openai_key)

# AI FUNCTIONS
def generate_blog(topic, additional_text):
    if not topic or not topic.strip():
        return "❗️Please enter a topic first."

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert blog writer. Write clear, engaging, well-structured posts "
                "with headings, short paragraphs, and actionable takeaways. Respond directly with the blog text."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Topic: {topic.strip()}\n\n"
                f"Additional guidance (optional): {additional_text.strip()}\n\n"
                "Please write a blog post with:\n"
                "- An H1 title\n- 3–5 concise sections with H2s\n- Bulleted tips where useful\n"
                "- A short conclusion with a call-to-action\n"
                "Target length: ~700–900 words."
            ),
        },
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_completion_tokens=900,
        temperature=0.7,
    )
    return (response.choices[0].message.content or "").strip()
      


# END AI FUNCTIONS

st.set_page_config(page_title="OpenAI API", layout="wide")

st.title("OpenAI API Webapp")
st.write("Hello from Streamlit 🚀")
st.sidebar.title("AI APP")

ai_app = st.sidebar.radio("Choose an AI App", ("Blog Generator", "Chatbot Commencal"))

if ai_app == "Blog Generator":
    st.header("Blog Generator")
    st.write("Input topic to generate a blog about it using OpenAI API")
    topic =  st.text_area("Topic", height = 15)
    additional_text = st.text_area("Additional text", height = 30)

    if st.button("Generate Blog"):
        with st.spinner("Generating blog..."):
            blog_text = generate_blog(topic, additional_text)

        st.text_area("Generated Blog", value=blog_text, height=700)
