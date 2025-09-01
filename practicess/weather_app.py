import requests
import pprint as PrettyPrinter



api_key = "your weather api key"


# data getting from weatherapi.com
def get_weather(location):

    url = f"https://api.weatherapi.com/v1/current.json?key={api_key}&q={location}" 
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while fetching weather data: {e}")
        return None
    data = response.json()
    print("Weather data for", location , "\n")
    PrettyPrinter.pprint(data["current"])


def temp(location):
    url = f"https://api.weatherapi.com/v1/current.json?key={api_key}&q={location}" 
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while fetching weather data: {e}")
        return None
    data = response.json()
    temp_c = data["current"]["temp_c"]
    temp_f = data["current"]["temp_f"]
    print(f"Temperature in {location}: {temp_c}°C / {temp_f}°F")

def humidity(location):
    url = f"https://api.weatherapi.com/v1/current.json?key={api_key}&q={location}" 
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while fetching weather data: {e}")
        return None
    data = response.json()
    humidity = data["current"]["humidity"]
    print(f"Humadity in {location} : {humidity}% ")

def main():
    print("_"*30)
    print("Welcome to Weather app")
    print("1. data")
    print("2. temp")
    print("3. Humidity")
    # print("4. condition")

    choice = input("Enter you choice: ")

    if choice == ("1") or choice == "data":
        location = input("Enter your country name : ")
        get_weather(location)
    elif choice == ("2") or choice == "temp":
        location = input("Enter your country name : ")
        temp(location)
    elif choice == ("3") or choice == "humidity":
        location = input("Enter your country name : ")
        humidity(location)
    else:
        print("invalid input. Please try again")
        



if __name__ == "__main__":
    main()
    


