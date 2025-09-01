# the user enter a choice then check if right so game win otherwise loss
print("Welcome to adventure game")
adven1 = input("Your are in forest there is no one helping you there are two ways to go 'right' and other is 'left' which way you want to go: ")

if adven1 == "left":
    adven2 = input("You reach to an empty chamber there is water you want to drink (yes/no): ")
    if adven2 == "yes":
        print("That water contain poison you LOSS")

    elif adven2 == "no":
        adven3 = input("There is a way to go you want to go (go/not): ")
        if adven3 == "go":
            print("You loss")
        else:
            print("You are eaten by an lions You loss")

else:
    adven4 = input("You are reach to an house there is stranger eating do you wanna eat with him (yes/no): ")
    if adven4 == "yes":
        print("You get an friend you win")
    else:
        print("you loss the stranger killed you.")