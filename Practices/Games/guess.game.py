import random

def guessing():
    guess_count = 0
    pipe = random.randint(1,100)

    while True:
      guess = int(input("Enter a number: "))

      if guess > pipe:    
            print("Lower number please!")
            guess_count += 1
      elif guess < pipe:
          print("Higher number please")
          guess_count += 1
      else:
          print(f"You attempt {guess_count} to find correct number {pipe}")
          break
          
          
    

guessing()