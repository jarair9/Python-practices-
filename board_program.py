import json
import string
import random
import os

file = "jsons/data.json"

# Ensure directory exists
os.makedirs(os.path.dirname(file), exist_ok=True)

# If the JSON file does not exist, create it as an empty list.
if not os.path.exists(file):
    with open(file, "w") as f:
        json.dump([], f, indent=4)


def generate_id(length=6):
    chars = string.ascii_letters + string.digits
    return ''.join(random.choices(chars, k=length))


def load_data():
    if not os.path.exists(file):
        return []

    with open(file, "r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            # If file is empty or contains invalid JSON, start fresh
            data = []

    return data


def save_data(data):
    with open(file, "w") as f:
        json.dump(data, f, indent=4)


def add_user():
    name = input("Enter student name: ").strip()

    while True:
        try:
            student_class = int(input("Enter student class: "))
            break
        except ValueError:
            print("Please enter a valid number for class.")

    while True:
        try:
            marks = int(input("Enter student marks: "))
            break
        except ValueError:
            print("Please enter a valid number for marks.")

    data = load_data()

    student = {
        "name": name,
        "Class": student_class,
        "marks": marks,
        "id": generate_id()
    }

    data.append(student)
    save_data(data)

    print("User added successfully!")


def return_data():
    search = input("Enter student name: ").strip()

    data = load_data()

    for user in data:
        if user["name"].lower() == search.lower():
            print("\nStudent found:")
            print(f"Name  : {user['name']}")
            print(f"Class : {user['Class']}")
            print(f"Marks : {user['marks']}")
            print(f"ID    : {user['id']}")
            return

    print("Student not found.")


def main():
    print("Welcome to board program project")
    print('Select:')
    print('1. Add student Data')
    print('2. Search User details')
    print('3. Quit')
    while True:
        print("Press ctrl + c for Quit..\n")

        setting = input('\nEnter your configuration (1 , 2 or 3): ').strip()

        if setting == '1':
            print('Add carefully ')
            add_user()
        elif setting == '2':
            print('Enter name correctly')
            return_data()
        elif setting == '3':
            break
        else:
            print("Invalid option. Please run the program again and choose 1 or 2.")


if __name__ == '__main__':
    main()
