import streamlit as st
from langchain_helper import get_few_shot_db_chain

# Page Configuration
st.set_page_config(
    page_title="AtliQ T-Shirts Q&A",
    page_icon="👕",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Styling
st.markdown("""
    <style>
    .main {
        background-color: #f6f8fa;
        padding: 2rem;
    }
    .stTextInput > div > div > input {
        font-size: 18px;
        padding: 10px;
    }
    .big-title {
        text-align: center;
        font-size: 40px;
        color: #1e3a8a;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #555;
        margin-bottom: 2rem;
    }
    .answer-box {
        background-color: #e0f7fa;
        padding: 20px;
        border-radius: 12px;
        font-size: 18px;
        color: #004d40;
        font-weight: 500;
        margin-top: 1.5rem;
        border-left: 5px solid #00796b;
    }
    .stButton>button {
        background-color: #007BFF;
        color: white;
        font-size: 18px;
        padding: 0.6em 1.2em;
        border-radius: 8px;
        margin-top: 10px;
        border: none;
        transition: 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    </style>
""", unsafe_allow_html=True)

# App Title
st.markdown('<div class="big-title">🧠 Urban T-Shirts: Smart Database Q&A</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Ask any question about your inventory, stock, or sales – get answers instantly!</div>', unsafe_allow_html=True)

# Input Field
question = st.text_input("🔍 Type your question here:")

# Button to Generate Answer
if st.button("🚀 Generate Answer"):
    if question.strip() == "":
        st.warning("❗ Please enter a question before clicking Generate.")
    else:
        chain = get_few_shot_db_chain()
        with st.spinner("🧠 Thinking..."):
            try:
                response = chain.invoke({"query": question})
                answer = response.get("result", "⚠️ No answer returned.")
                st.markdown(f'<div class="answer-box">✅ {answer}</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"⚠️ Error: {e}")
# Footer
st.markdown('<div class="footer">© 2025 Adarsh Korade | Powered by LangChain & Streamlit</div>', unsafe_allow_html=True)
