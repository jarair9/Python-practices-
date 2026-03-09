import json
import os
def save_data():
    # Initialize an empty dictionary
    data = []
    # Get the key input from the user
    # Get the value input from the user
    key = input("Enter the key: ")
    value = input("Enter the value: ")
    entry = {key: value}

   

    # Store the input as key-value pair in the dictionary
    data.append(entry)
    
    # Save the dictionary to a JSON file
    with open('mablib/data.json', 'a') as f:
        f.write(json.dumps(data) + '\n')

    print("Data saved successfully.")


save_data()