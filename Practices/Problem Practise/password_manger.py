# the user entera password that pass is stored in a file then if user want to veiw the pass so he can view or add


def add():
    name = input("Enter name : ")
    password = input("Enter you password: ")
    print("'Your password has been saved'")

    with open("pass.txt", "a") as f:
        f.write(f"name: {name} | password : {password}\n")


def view():
    with open("pass.txt", "r") as f:
        content = f.read()
        print("\nYour name and passwords are saved and secure here. ")
        print(content)
    
def main():
    while True:
        print("you are in password manager app")
        print("1. Add")
        print("2. View")
        print("3. Exit")
        user_one = input("Enter what would like to do: ").lower().strip()

        if user_one == 1 or user_one == "add":
            add()
        elif user_one == 2 or user_one == "view":
            view()
        elif user_one == 3 or user_one == "exit":
            print("Goodbye")
            break
        else:
            print("Invalid input")

main()

    


