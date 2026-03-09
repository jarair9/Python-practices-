
import speech_recognition as sr
import pygame
import google.generativeai as genai
import edge_tts
import json
import asyncio
import pyttsx3
import os
import time
import random
import playsound
from rapidfuzz import fuzz



# add your api key in config.json file. the variable is API_KEY = "your api"

pygame.mixer.init()
r = sr.Recognizer()

with open("config.json") as f:
    config = json.load(f)

api_key = config["API_KEY"]
genai.configure(api_key=api_key)       
model = genai.GenerativeModel(model_name="models/gemini-2.0-flash") # other models are paid

# if api key not found
if "API_KEY" not in config:
    raise KeyError("API_KEY missing in config.json")


music_folder = "music"
music_files = [f for f in os.listdir(music_folder) if f.endswith(".mp3")]

english_urdu_dict = {
    "apple": "seb",
    "book": "kitaab",
    "water": "pani",
    "love": "mohabbat",
    "peace": "aman",
    "friend": "dost",
    "light": "roshni",
    "dark": "andhera",
    "happy": "khushi",
    "sad": "udaasi",
    "house": "ghar",
    "school": "school",
    "time": "waqt",
    "work": "kaam",
    "money": "paise",
    "heart": "dil",
    "beautiful": "khoobsurat",
    "life": "zindagi",
    "death": "maut",
    "truth": "haqeeqat",
    "man": "aadmi",
    "woman": "aurat",
    "father": "walid",
    "mother": "walida",
    "brother": "bhai",
    "sister": "behen",
    "day": "din",
    "night": "raat",
    
}
# play music 
def play_random_song():
    song = random.choice(music_files)
    song_path = os.path.join(music_folder, song)
    pygame.mixer.music.load(song_path)
    pygame.mixer.music.set_volume(0.4)
    pygame.mixer.music.play()
    print(f"🎵 Now playing: Music")

pygame.mixer.music.set_endevent(pygame.USEREVENT)
play_random_song()

# Pytts model
def pytts(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
# edgetts model
def speak(text):
    async def run():
        tts = edge_tts.Communicate(text, voice="en-US-AriaNeural")
        await tts.save("voice.mp3")
    asyncio.run(run())
    playsound.playsound("voice.mp3")
    os.remove("voice.mp3")

# optional function
def typer(text,delay=0.03):
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)

# Recognize Audio of user    
def listenr():
    with sr.Microphone() as source:
        print("🎤 Listening...")
        r.adjust_for_ambient_noise(source)
        try:
            audio = r.listen(source, timeout=6)  # timeout here, not below
        except sr.WaitTimeoutError:
            print("⏱️ You didn’t speak in time.")
            return ""

        try:
            text = r.recognize_google(audio, language="en")  # use 'ur' for Urdu
            print(f"📝 You said: {text}")
            return text
        except sr.UnknownValueError:
            print("❓ I couldn't understand you.")
        except sr.RequestError as e:
            print(f"⚠️ API error: {e}")
        return ""

# Asking question from user
def text():
    all_urdu_words = list(english_urdu_dict.values())

    for english_word in english_urdu_dict:
        correct_answer = english_urdu_dict[english_word]
        hint = [correct_answer]

        other_words = [word for word in all_urdu_words if word != correct_answer]
        hint += random.sample(other_words, 3)  

        random.shuffle(hint)
        print("Hint :", hint)

        user_answer = input(f"What is the meaning of '{english_word}' in Urdu?:  ")

        if user_answer.strip().lower() == correct_answer:
            print("✅ Correct!\n")  
        else:
            print(f"❌ Incorrect! The correct answer is: {correct_answer}\n")

# ask Questions from user

# Speaking AI
def talkai():

    print("🤖 Starting your AI Language learning model...")

    history = [
    {"role": "user", "parts": [f"You're a helpful and casual assistant. Speak like a friend. No lists or formal abilities. don,t use ** or other symbols just text. Also correct the gramer of the user. you are a english teacher and you have to analyse user response and give a response in one paraghraph no heading just words."]},
    {"role": "user", "parts": ["Hello!"]},]

    while True:
        
        response = model.generate_content(history)
        response_text = response.candidates[0].content.parts[0].text

        print(f"AI: {response_text}")
        pytts(response_text)

        print("🎤 Listening for your response...")
        user_input = listenr()
        if user_input.lower() in ["exit", "quit", "stop"]:
            pytts("Goodbye! Have a nice day.")
            break
        
        history.append({"role": "model", "parts": [response_text]})
        history.append({"role": "user", "parts": [user_input]})
   
# AI chatbot
def chatai():

    print("Initializing AI...")


    while True:
        user = input("YOU: ")
        response = model.generate_content("You're a helpful and casual assistant. Speak like a friend. No lists or formal abilities. don,t use ** or other symbols just text. Also correct the gramer of the user.")
        response_text = response.candidates[0].content.parts[0].text
        print(f"AI: {response_text}")
        pytts(response_text)
        if user.lower() in ["exit", "quit", "stop"]:
            pytts("Goodbye! Have a nice day.")
            break



def main():
    
    print("\n📚 Welcome to our language learning model")
    pytts("Welcome to our language learning model")

    print("If you are a complete beginner in English you can learn English here")
    pytts("If you are a complete beginner in English you can learn English here")

    print("Now you have to select you level")
    pytts("Now you have to select you level")

    print("1. Beginner")
    pytts("number 1. beginner")

    
    print("2. Master")
    pytts("number 2. Master")

    pytts("Select you English conversation level")

    user_choice = input("Select you English conversation level: ")

    if user_choice in ["1", "beginner"]:
        print("In case you are beginner so you need to learn some words in English")
        speak("In case you are beginner so you need to learn some words in English")

        for words in english_urdu_dict:
          print(f"{words} => {english_urdu_dict[words]}")

        print("\nWhen AI ask for a Question You will tell the answer")
        speak("When AI ask for a Question You will tell the answer")
        text()

    # elif user_choice in ["2", "Intermediate", "intermediate"]:
    #     print("You are intermidiate so i will test you English ")
    #     speak("You are intermidiate so i will test you English ")

    #     print("Answer some of questions and test your self.")
    #     speak("Answer some of questions and test your self.")

    elif user_choice in ["3", "master"]:
        print("Since you're a master, you'll speak with AI to correct your grammar.")
        speak("Since you're a master, you'll speak with AI to correct your grammar.")


        talkai()
        text()
    else:

        print("❌ Invalid choice.")
        speak("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()  

    

            













            
                
                    
            
            
                
                