# creating password checker 
# strong password contains uppercase ,lowercase, numbers,symbols and atleast one capital letter.
from colorama import Fore
import string
def checker():
    user = input("Enter your password: ")
    symbols = ["@","#","$","%","^","&","*"]
    upercase = list(string.ascii_uppercase)
    lowercase = list(string.ascii_lowercase)
    numbers = list(string.digits)
    length = 8

    if len(user) < length:
        print(Fore.RED + "Password must be at least 8 characters long.")
        return False
    if not any(char in symbols for char in user):
        print(Fore.RED + "Password must contain at least one symbol.")
        return False
    if not any(char in upercase for char in user):
        print(Fore.RED + "Password must contain at least one uppercase letter.")
        return False
    if not any(char in lowercase for char in user):
        print(Fore.RED + "Password must contain at least one lowercase letter.")
        return False
    if not any(char in numbers for char in user):
        print(Fore.RED + "Password must contain at least one number.")
        return False
    print(Fore.GREEN + "Password is strong.")
    return True

if __name__ == "__main__":
    print(Fore.YELLOW + "Welcome to the Strong Password Checker!")
    while True:
        if checker():
            break
        else:
            print(Fore.YELLOW + "Please try again.")




    