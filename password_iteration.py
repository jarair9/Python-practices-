# # the brute force password cracker






from itertools import product

def brute_force_password(target_password, charset, max_length):
    """
    target_password: the password string to find
    charset: string of characters to use, e.g. "abc123"
    max_length: maximum length of passwords to try
    """
    attempts = 0

    for length in range(1, max_length + 1):
        for combo in product(charset, repeat=length):
            guess = ''.join(combo)
            attempts += 1

            if guess == target_password:
                return guess, attempts  # found it!

    return None, attempts  # not found within this length


# ===== Example use =====
user_password = input("Enter a password (for demo, keep it short!): ")
charset = "abcdefghijklmnopqrstuvwxyz0123456789"
max_len = 10 # be careful: grows VERY fast

found, tries = brute_force_password(user_password, charset, max_len)

if found:
    print(f"Password found: {found} in {tries} attempts.")
else:
    print(f"Password not found within length {max_len}. Tried {tries} combinations.")



from itertools import product

password = input("Enter a password: ")

characters = "abcdefghijklmnopqrstuvwxyz0123456789"   # characters to try
max_length = 10          # try passwords up to length 3

for length in range(1, max_length + 1):
    for combo in product(characters, repeat=length):
        guess = "".join(combo)
        print(guess)

        if guess == password:
            print("Password found!", guess)
            exit()

print("Password not found.")



# Brute force Three
from itertools import product
import time


start = time.time()
password = str(input("Enter you password: "))
chars = "abcdefghijklmnopqrstuvxyz"
checks = 0
repeats = 1000000


for lenght in range(1,7):
    for guess in product(chars,repeat=lenght):
        checks +=1
        attempts = "".join(guess) # How it uses join here ?
    if attempts != password:
        print(f"checks : {checks}")
        
    if attempts == password:
        print(password)
