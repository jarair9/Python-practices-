import requests
import prettyprinter as pp



base_url = "https://v6.exchangerate-api.com/v6/latest/USD"

# Function to get the latest currency exchange rates
# def database():
    
#     api_key = "bec1a04a919c9d55a7ea9d30"    
#     endpoint = f"https://v6.exchangerate-api.com/v6/{api_key}/latest/USD"
#     data = endpoint
#     response = requests.get(data)
#     pp.pprint(response.json())
    

# Gives list of currencies
def list_of_currency():
    
    api_key = "bec1a04a919c9d55a7ea9d30"    
    endpoint = f"https://v6.exchangerate-api.com/v6/{api_key}/latest/USD"
    data = endpoint
    response = requests.get(data)
    for key , value in enumerate(response.json()["conversion_rates"]):
        print(f"{key}: {value}")


# takes prices of each currency
def currency_price():
    api_key = "bec1a04a919c9d55a7ea9d30"    
    endpoint = f"https://v6.exchangerate-api.com/v6/{api_key}/latest/USD"
    data = endpoint
    response = requests.get(data)
    for key in response.json()["conversion_rates"]:
        print(f"{key}: {response.json()['conversion_rates'][key]}")

def convert_currency(amount,to_corrency):
    try:
        api_key = "bec1a04a919c9d55a7ea9d30"
        base = "USD"
        url = f"https://v6.exchangerate-api.com/v6/{api_key}/latest/{base}"
        response = requests.get(url)
        data = response.json()
        rate = data["conversion_rates"].get(to_corrency.upper())
        convertion = amount * rate
        if rate:
            print(f"{amount} {base} = {convertion:.2f} {to_corrency}")
        else:
            print("invalid currency")
    except Exception as j:
        print(f"An error occurred: {j}")
# checks the conversion rate between two currencies
def c_to_c(c1,c2):
    api_key = "bec1a04a919c9d55a7ea9d30"
    url = f"https://v6.exchangerate-api.com/v6/{api_key}/latest/USD"
    response = requests.get(url)
    data = response.json()

    rate = data["conversion_rates"]

    if c1 not in rate and c2 not in rate:
        print("invalid currency")
        return
    
    convertion_rate = rate[c2] / rate[c1]

    print(f"1 {c1} = {round(convertion_rate,2)} {c2}")

def main():
    print("____________________")
    print("1. List of currencies")
    print("2. Currency price")
    print("3. Convert currency")
    print("4. Currency to currency")
    print("5. Exit")

    while True:
        choice = input("Enter you choice: ")
        if choice == "list of currencies" or choice == "1":
            list_of_currency()
            
        elif choice == "currency price" or choice == "2":
            currency_price()
        elif choice == "convert currency" or choice == "3":
            amount = float(input("Enter the amount: "))
            to_currency = input("Enter the currency to convert to: ").upper()
            convert_currency(amount, to_currency)
        elif choice == "currency to currency" or choice == "4":
            c1 = input("Enter a pair 1: ").upper().upper()
            c2 = input("Enter a pair 2: ").upper().upper()
            c_to_c(c1,c2)
        elif choice == "exit" or choice == "5":
            return
        else:
            print("invalid input")

if __name__ == "__main__":
    main()


    

    
    

