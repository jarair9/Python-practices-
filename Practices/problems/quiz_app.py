import time
import pygame
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
    },
    3: {
        "question": "What is the chemical symbol for gold?",
        "options": ["Au", "Ag", "Fe", "Cu"],
        "answer": "Au"
    },
    4: {
        "question": "Who wrote the play 'Romeo and Juliet'?",
        "options": ["William Shakespeare", "Charles Dickens", "Jane Austen", "Mark Twain"],
        "answer": "William Shakespeare"
    },
    5: {
        "question": "What is the capital of Japan?",
        "options": ["Tokyo", "Beijing", "Seoul", "Shanghai"],
        "answer": "Tokyo"
    },
    6: {
        "question": "What is the smallest prime number?",
        "options": [1, 2, 3, 4],
        "answer": 2
    },
    7: {
        "question": "What is the largest ocean on Earth?",
        "options": ["Atlantic Ocean", "Pacific Ocean", "Indian Ocean", "Arctic Ocean"],
        "answer": "Pacific Ocean"
    },
    8: {
        "question": "What is the capital of Australia?",
        "options": ["Sydney", "Melbourne", "Canberra", "Brisbane"],
        "answer": "Canberra"
    },
    9: {
        "question": "What is the chemical symbol for oxygen?",
        "options": ["O", "H", "N", "C"],
        "answer": "O"
    },
    10: {
        "question": "Who discovered the theory of relativity?",
        "options": ["Isaac Newton", "Albert Einstein", "Galileo Galilei", "Niels Bohr"],
        "answer": "Albert Einstein"
    }
}



def typer(text,delay=0.04):
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
    print()

    # pygame.mixer.init()
    # pygame.mixer.music.load("files/track.mp3")
    # for char in text:
    #     if char == ":":
    #         pygame.mixer.music.stop()
    #     else:
    #         pygame.mixer.music.play()
        
def gt_questions(questions):
    score = 0
    for question in questions:
        print()
        typer(questions[question]["question"])

        for option in questions[question]["options"]:
            typer(option)
        answers = input("Enter your choice: ").strip().capitalize()
        if answers == questions[question]["answer"]:
            typer("Correct answer ")
            score += 1 
        else:
            typer(f"Incorrect answer ")
        print()
    typer(f"Your score in quiz is {score} in 10 Questions")
        
        
gt_questions(questions)