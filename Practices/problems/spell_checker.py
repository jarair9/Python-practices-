from spellchecker import SpellChecker
from colorama import Fore

spell = SpellChecker()
def corrected(text):
    words = text.split()
    
    for word in words:
        correct = spell.correction(word)
        if word != correct:
            print(Fore.RED,correct,end="",flush=True)
            # print(F"\nYou Entered a wronge words: {correct}",end="",flush=True)
        else:
            print(Fore.GREEN,word,end="",flush=True)

def candi(text):
    words = text.split()
    for word in words:
        correct = spell.correction(word)
        if word != correct:
            candit = spell.candidates(word)
            
            print(Fore.YELLOW,candit,end="",flush=True)
            print()
            # print(F"\nYou Entered a wronge words: {correct}",end="",flush=True)
        else:
            print(Fore.GREEN,word,end="",flush=True)

def main():
    print("Welcome to Ahmad programer word checker")
    print()
    text = str(input("Enter your sentence for words checks: "))  
    corrected(text)  
    print()    
    # candi(text)
if __name__ == "__main__": 
    main()