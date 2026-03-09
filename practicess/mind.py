
import random

class mind:
    def __init__(self,words,guess):
        self.words = words # List of words
        self.guess = guess # user guess matching

    def gen_words(self):
        word = random.choice(self.words)
        return word
    def get_guess(self):
        guess = input("Enter your guess: ")
        if guess in self.words:
            print("Correct guess!")
        else:
            print("Incorrect guess. Try again.")
    def main(self):
        print("Welcome to the word guessing game!")
        while True:
            word = self.gen_words()
            self.get_guess()
            if input("Do you want to play again? (yes/no): ").lower() != 'yes':
                break
        print("Thanks for playing!")

    word = ['apple', 'banana', 'cherry', 'date', 'elderberry']

m = mind(words=mind.word, guess=None)
m.main()