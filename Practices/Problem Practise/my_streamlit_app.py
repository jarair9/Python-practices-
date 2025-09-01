import streamlit as st
import google.generativeai as genai

# Set your Gemini API key
api_key = "AIzaSyCphPLGWXvbNMQlPy5WXFrbF_6E_Em721U"
genai.configure(api_key=api_key)





# Load the Gemini model
model = genai.GenerativeModel(model_name="models/gemini-1.5-pro")

# UI Layout
st.set_page_config(page_title="Gemini Chat", layout="centered")
st.title("🤖JarairGPT")
st.subheader("Chat with AI or get summaries")

# Input text
user_input = st.text_input("✏️ Enter your text or question:")

# Chat memory (optional)
if "history" not in st.session_state:
    st.session_state.history = []

# When user types input
if user_input:
    prompt = (
        f"You are a helpful assistant. Analyze the following input and give a short summary "
        f"in one paragraph. If it's a question, give a short and concise answer. "
        f"Talk like a friend if the tone is casual.\n\nInput: {user_input}"
    )

    try:
        with st.spinner("Thinking..."):
            response = model.generate_content(prompt)
            st.session_state.history.append((user_input, response.text))
            st.success("Response ready!")

    except Exception as e:
        st.error(f"⚠️ An error occurred: {e}")

# Display previous messages
if st.session_state.history:
    st.markdown("### 💬 Conversation History")
    for i, (q, a) in enumerate(reversed(st.session_state.history), 1):
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Gemini:** {a}")
        st.markdown("---")









