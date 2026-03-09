# the user deposit the money 
# - user bett the money on game
# - if win update and if not loss
# - user can withdraw the money
import random
import time
balance = 0

def type(text,delay=0.04):
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
    print()

def deposit():
    global balance
    dep = int(input("\nEnter the amount that you like to deposit : "))
    balance += dep
    type(f"Your account balance is {balance}$")

def bet():
    global balance
    want = input("Do you want to play a bet (y/n): ")
    if "y" not in want:
        type("Ok let,s play next time")
        return
    bet1 = int(input("How much you would like to bet on $: "))
    
    if bet1 > balance or bet1 <= 0:
        type("Enter a valid amount")
        exit()
    balance -= bet1
    type(f"Your remaining balance is {balance}$")
    while True:
        if "y" in want:
            outcome = random.choice(["🔔 🔔 🍋","💣 🐷 🧨","💎 🍀 🍊","7️⃣ 7️⃣ 7️⃣","🍒 🍒 🍒"])
            type("Your betting is starting.....")
            type(f"bet {outcome}")
            if outcome == "7️⃣ 7️⃣ 7️⃣":
                winning = bet1*5
                balance +=winning
                type("congrates you win ")
                type(f"You get {winning}$")
                print(balance)
            elif outcome == "🍒 🍒 🍒":
                winning = bet1*2
                balance+=winning
                type("congrates you win")
                type(f"you win {winning}$")
                print(balance)
            else:
                type("You loss")
                type(f"you balance is {balance}$")

            want2 = input("Type 'p' to try again, or anything else to quit: ")

            if want2 != "p":
                type("Thanks for playing!")
                return
            if balance == 0:
                type("You are out of money")
                type("If you want to play you need to deposit again")
                d2 = input("Do you want to deposit again (y/n): ")
                if "y" not in d2:
                    type("Ok then let,s play next once you deposit ")
                    exit()
                else:
                    deposit()
            
            bet2 = int(input("How much you would like to bet on $: "))
            if bet2 > balance or bet2 <= 0:
                type("Enter a valid amount")
                continue
            
            
            balance = balance-bet2

            type(f"Your remaining balance is {balance}$")
            
def main():
    type("Welcome to Royal Casino game")
    type("Currently we have only betting section but soon we will launch an amazing features")
    type("Let,s begin")
    deposit()
    type("Your payment will be safe here")
    bet()



if __name__ == "__main__":
    main()

