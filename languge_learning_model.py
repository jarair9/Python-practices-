import pyttsx3
import speech_recognition as sr
from mistralai.client import MistralClient
import os
import sys

# Keep the module reference `sr` and create a separate Recognizer instance
recognizer = sr.Recognizer()

MISTRAL_API_KEY="i0ICrPypIZ0raJVUky81Vu5DRZhIiP6m"

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


def recognize_speech():
    with sr.Microphone() as source:
        print('Listening...')
        audio = recognizer.listen(source,timeout=5)
        try:
            # text = recognizer.recognize_bing(audio,key=None,language="en-US")
            text = recognizer.recognize_google(audio)
            print(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            print("Could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
            return None
        

def AI():
    client = MistralClient(api_key=MISTRAL_API_KEY)
    user = recognize_speech()
    if user is None:
        return "Sorry, I didn't catch that. Could you please repeat?"
    try:
        response = client.chat(
            model="mistral-small-latest",
            messages=[
                {"role": "system", "content": "You are an ai language learning assistant that helps users practice and improve their language skills through conversation.Ask questions, provide corrections, and offer explanations as needed.keep answers short ,concise and engaging."},
                {"role": "user", "content": user}
            ],
        )
    except Exception as e:
        return f"AI is Busy right now: {e}"

    return response.choices[0].message.content

def main():
    print("Language Learning AI is ready to chat!")
    print("-"*30)
    while True:
        response = AI()
        print(f"AI: {response}")
        speak(response)

if __name__ == "__main__":
    main()
