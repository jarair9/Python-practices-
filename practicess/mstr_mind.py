import random

length = 4
colours = ["B", "Y", "R", "W", "P"]
tries = 10

# Generates a random code of given length
def coder():
    return [random.choice(colours) for _ in range(length)]

# Compares guess with code
def compare(guess, code):
    guess = list(guess)
    exact = 0
    partial = 0

    # Make copies to modify
    code_copy = code.copy()
    guess_copy = guess.copy()

    # First pass: check for exact matches
    for i in range(length):
        if guess[i] == code[i]:
            exact += 1
            code_copy[i] = None
            guess_copy[i] = None

    # Second pass: check for correct color but wrong position
    for i in range(length):
        if guess_copy[i] is not None and guess_copy[i] in code_copy:
            partial += 1
            code_copy[code_copy.index(guess_copy[i])] = None

    print(f"Exact (right color and position): {exact}")
    print(f"Partial (right color, wrong position): {partial}")
    return exact, partial

# Example usage
code = coder()
print("Secret code:", code)  # For testing purposes only
compare("BWRY", code)




    


    
   

        