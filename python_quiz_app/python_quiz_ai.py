# making an advance ai model project for python Quiz
# first we make a client that can send requests to the server
# then write a code that takes a question.

import google.generativeai as genai
import ast
import json
from dotenv import load_dotenv
import os
import time
from colorama import Fore
load_dotenv()

api_key= os.getenv("API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel(model_name="models/gemini-2.0-flash")

# Prompt
response = model.generate_content("""
        You are a Python expert. Please generate Python quiz questions, each with multiple-choice options and the correct answer.
        Format each question as a Python dictionary, for example:
        questions = {
            1: {
                "question": "What is the capital of France?",
                "options": ["Paris", "London", "Berlin", "Madrid"],
                "answer": "Paris"
            },
            2: {
                "question": "What is the largest planet in our solar system?",
                "options": ["Earth", "Mars", "Jupiter", "Saturn"],
                "answer": "Jupiter"
            }
        }
        Do not use (```python),(```json) and  (```) at start and end of response.
        Do not include any headings, explanations, or decorative characters. Only provide the questions in the specified dictionary format and also do not use python word at starting
        Do not include any additional text or comments, just the dictionary with questions.
""")

raw_text = response.text
wrong_answer = 0

# Clean code block markers if present
for marker in ["```json", "```python", "```"]:
    if raw_text.startswith(marker):
        raw_text = raw_text.replace(marker, "").strip()
    if raw_text.endswith("```"):
        raw_text = raw_text[:-3].strip()

# Convert to Python dict
questions_dict = ast.literal_eval(raw_text)

# Save as JSON
with open("python_quiz_app/python_quiz.json", "w") as f:
    json.dump(questions_dict, f, indent=4)
    
# initial phrases
print(Fore.GREEN +"\nWelcome to the Python Quiz!"+ Fore.RESET)
print(Fore.GREEN+"Ai powered Python quiz questions "+Fore.RESET)
print(Fore.GREEN+"Type your correctly from options "+Fore.RESET)
print(Fore.YELLOW+"Use your mouse for copy option for answer if you don,t wanna type.\n"+Fore.RESET)

# Taking question option from .json file
# item[0] is the question number
# item[1] is the question dictionary
for item in questions_dict.items():
    print()
    print(Fore.GREEN +f"Question {item[0]}: {item[1]['question']}"+ Fore.RESET)
    for option in item[1]['options']:
        print(Fore.YELLOW + f"- {option}" + Fore.RESET) # print(Fore.YELLOW + f"Option: {option}" + Fore.RESET) # print(Fore.YELLOW+"Options:", ", ".join(item[1]['options'])+ Fore.RESET)
    correct_ans =  item[1]['answer']
    start = time.time()

    quiz_ans = input("Your answer: ")

    if quiz_ans.lower() == correct_ans.lower():
        print(Fore.GREEN+"Correct!"+Fore.RESET)
    else:
        print(Fore.RED+f"Wrong! The correct answer is: {correct_ans}"+Fore.RESET)
        wrong_answer += 1

    print()
    
elapsed = time.time() - start

print(f"Time taken for this question: {elapsed:.2f} seconds")
print(f"You got {len(questions_dict) - wrong_answer} out of {len(questions_dict)} questions correct.")
print(f"Quiz completed! in {elapsed:.2}")





    







    
        

    


        
         




    

    
    