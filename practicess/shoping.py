import json
from colorama import Fore
    
def ad_data(data):
    try:
        with open("data2.json", "r") as f:
            data_list = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data_list = []
    data_list.append(data)
    with open("data2.json", "w") as f:
        json.dump(data_list, f, indent=4)


def electronic(user):
    products = ["hairdryer", "Carpetcleaner", "Plug", "Bulbs", "Washingmachine"]
    print(Fore.YELLOW + "Welcome to the Electronics Shop")
    print(Fore.CYAN + "Select the items from below:\n")
    for idx, words in enumerate(products, 1):
        print(Fore.GREEN, f"{words}", end="   ", flush=True,)
        if idx % 3 == 0:
            print()

    menu = {
        "hairdryer": 100,
        "Carpetcleaner": 200,
        "Plug": 10,
        "Bulbs": 5,
        "Washingmachine": 500
    }
    # user = str(input(Fore.YELLOW + "\nWhat would you like to buy: "))
    if user not in menu:
        print(Fore.RED + "Item not found. Please try again.")
        return False
    if user in menu:
        price = menu[user]
        print(Fore.YELLOW + f"\nThe price of {user} is {price}$.")
        confirm = input(Fore.CYAN + "Do you want to buy it? (yes/no): ").strip().lower()
        if confirm == "yes":
            print(Fore.GREEN + f"You have successfully bought {user} for {price}$.")
            data = {
            "user": user,
            "product": user,
            "price": price
            }
            ad_data(data)
        elif confirm == "no":
            print(Fore.RED + "Purchase cancelled.")
            return False
        else:
            print(Fore.RED + "Invalid input. Please enter 'yes' or 'no'.")
            return 

    
    
    
