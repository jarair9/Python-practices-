import json
import os
from colorama import Fore

def check(name):
    with open("data.json", "r") as f:
        data_for = json.load(f)
        content = data_for
    for item in content:
        if name in item:
            try:
                num = int(item.split(":")[1]) # it remove the ":" and convert the string to integer and take the second item from the list.
                if num >= 450:
                    print(Fore.GREEN,f"The student {name} has {num} marks"+Fore.RESET)
                else:
                    print(Fore.CYAN,f"The student {name} has {num} marks"+Fore.RESET)
            except ValueError:
                print("\nInvalid syntex is used for marks in database\n")


            
            

def add_data(name,marks):
    data_list = []
    if os.path.exists("data.json"):
        with open("data.json", "r") as f:
            try:
                data_list = json.load(f)  # ✅ Correct: load from file
            except json.JSONDecodeError:
                data_list = []  # If file is empty or invalid JSON
    
    # Append new entry
    data = f"{name} : {marks}"
    data_list.append(data)

    # Write updated list back to file
    with open("practices/problems/data.json", "w") as f:
        json.dump(data_list, f, indent=4)      
            

def main():
    print("Enter q for quit :)")
    print("_"*30)
    print("Welcome to kpk board program")
    print("Select from below to enter in database")
    print("1. Add data")
    print("2. Check data\n")
    choice = input("Enter your choice: ").strip().lower()
    while True:
        if choice == "1" or choice == "add data":
            # print(Fore.GREEN,"Only enter name and marks of the student"+ Fore.RESET)
            

            name = str(input("Enter a name: ")).strip()
            if name == "" or name == " ":
                print(Fore.RED,"Name cannot be empty"+Fore.RESET)
                continue
            elif len(name) < 3:
                print(Fore.RED,"Name must be at least 3 characters long"+Fore.RESET)
                continue
            elif not name.isalpha():
                print(Fore.RED,"Name must contain only alphabets"+Fore.RESET)
                continue
            if name.lower() == "q":
                return
            marks = int(input("Enter a marks: "))
            if marks < 0:
                print(Fore.RED,"Marks must be between 0 and 600"+Fore.RESET)
                continue
            elif not str(marks).isdigit():
                print(Fore.RED,"Marks must be a number"+Fore.RESET)
                continue
            elif marks > 600:
                print(Fore.RED,"Marks must be between 0 and 600"+Fore.RESET)
                continue
            if str(marks).lower() == "q":
                print(Fore.GREEN+"Successfully added to database"+ Fore.RESET)
                return
            print()

            add_data(name,marks)
            
        elif choice == "2" or choice == "check data":
            name = input("Enter a name to check :").strip()
            if name == "" or name == " ":
                print(Fore.RED,"Name cannot be empty"+Fore.RESET)
                continue
            elif len(name) < 3:
                print(Fore.RED,"Name must be at least 3 characters long"+Fore.RESET)
                continue
            elif not name.isalpha():
                print(Fore.RED,"Name must contain only alphabets"+Fore.RESET)
                continue
            elif name.lower() == "q":
                return
            with open("data.json", "r") as f:
                content = f.read()
                if name.lower() and name.upper() not in content.lower() and content.upper():
                    print(Fore.RED,f"The student {name} is not in database"+Fore.RESET)
                    continue
            check(name)
            
        else:
            print(Fore.RED,f"Invalid input")
            break

if __name__ == "__main__":
    main()
   


        


    

       