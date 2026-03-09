# Little bit github code use
import turtle
import time 
import random
width , height = 700,600
COLORS =  ["red","blue","green","yellow","cyan","black","pink","purple","orange"]

def get_race():
    racer = 0
    while True:
        racer = input("Enter a number of turtles (2 - 10) : ")
        if racer.isdigit():
            racer = int(racer)
        else:
            print("input must be numaric.....")
            continue

        if 2 <= racer <= 10:
            return racer
        else:
            print("The input must i range of 1 - 5  ")





def create_turtle(colors):
    spacing = width // (len(colors)+ 1)
    turtles = []
    for i , color in enumerate(colors):
        racer = turtle.Turtle()
        racer.shape("turtle")
        racer.color(color)
        racer.left(90)
        racer.penup()
        x = -width //2 + (i + 1) * spacing
        racer.setpos(x,-height // 2 + 20)
        racer.pendown()
        turtles.append(racer)
    return turtles


def race(colors):
    turtles = create_turtle(colors)
    while True:
        for racer in turtles:    
            distance = random.randrange(1,20)
            racer.forward(distance)
            x, y = racer.pos()
            if y >= height // 2 - 10:
                return colors[turtles.index(racer)]

def init_screen():
    screen = turtle.Screen()
    screen.setup(width,height)
    screen.title("jarair")


num_racer = get_race()
init_screen()
random.shuffle(COLORS)
colors = COLORS[:num_racer]


winners = race(colors)
print(f"the winner of the race is {winners} ")
time.sleep(5)

