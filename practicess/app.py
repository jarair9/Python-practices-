import streamlit as st
import google.generativeai as genai



api_key="Your api key here"  # Replace with your actual API key
if not api_key:
    st.error("Please set your API key in the code.")
    st.stop()

genai.configure(api_key=api_key)
model = genai.GenerativeModel(model_name="models/gemini-2.0-flash")

def gpt():
    st.title("Chatgpt")
    st.write("Ask anything")
    user = st.text_input("Type your question here:")
    if st.button("Enter") and user:
        response = model.generate_content(
            f"you are a helpful assistant. Please assist with: {user}. Respond in short points or a paragraph."
        )
        st.write(response.text if hasattr(response, 'text') else response)



gpt()
    

   
    


