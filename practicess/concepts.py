a = 3
b = a
c = 3

print(a is b)
print(a is c)


a = [1,2,3]
b = a
c = [1,2,3]

print(a is b) # True
print(a is c) # why False

v = 45

def checks():
    v = 46 # We can also use global to inter global varaible into function.
    print("local variable",v)
checks()
print("global varaible",v)

f = 8//3 # Floor division gives round number if in float
print(f)
f = 8/3 # Division gives num in float
print(f)
f = 8 % 3 # remainder 
print(f)
f = 8 ** 3 # square root
print(f)
try:
    a = 5
    b = "2"
    print(a+b)
except Exception as e:
    print(e)

# a = set(input("Enter a numbers ")) # we can use all data types here like : dict, float,set,tuple,list,boolean etc.
# print(a)

x = 23
print(x)
x = "harry"
print(x)

# if you want to randomly select 4 int or str from list then this code is for you.
import random
COLORS = ["G", "W", "B", "Y", "O", "R"]
code = []
ran = random.choice(COLORS)
for _ in range(4):
    code.append(random.choice(COLORS))
print(code)


import os
try:
    os.remove("p1.py")
    print("remove successfully")
except FileNotFoundError:
    print("File not found")

print(True and False)
print(False and True)  

# os.system("shutdown /s /t 1")
print(2 + 2 * 2)

# The set() can not be iterates 
b = set({"harry", "kingdom", "king", "queen"})

for i in b:
    print(i)
w = {"harry": "king",
     "king": "harry"}
v = [ "harry", "kitaab"]

for i in v:
    print(i)

def text():
    value = []

    for i in range(len(v)):
        for words in v:
            hint = list(w.values())[:i+2]
            value.append(hint)
            print(value)
text()

english_urdu_dict = {
    "apple": "seb",
    "book": "kitaab",
    "water": "pani",
    "love": "mohabbat",
    "peace": "aman",
    "friend": "dost",
    "light": "roshni",
    "dark": "andhera",
    "happy": "khushi",
    "sad": "udaasi",
    "house": "ghar",
    "school": "school"
    }
for i in range(len(english_urdu_dict)):
    print(i)
    
# def text():
#     for i in range(len(english_urdu_dict)):
#         hint = list(english_urdu_dict.values())
#     print(hint)
#     for words in english_urdu_dict:
#         answer = input(f"What is mean by {words} in Urdu: ")
#         if answer == english_urdu_dict[words]:print("Correct")
#         else: print("incorrect")
#     i +=1
import random

# Dictionary of English to Urdu
english_urdu_dict = {
    "apple": "seb",
    "book": "kitab",
    "car": "gaari",
    "sun": "sooraj",
    "tree": "darakht",
    "moon": "chaand",
    "water": "pani",
    "fire": "aag",
    "river": "darya",
    "star": "sitara",
    "sky": "aasman",
    "earth": "zamin"
}
water = " A disability carer, who has pleaded guilty to causing a crash which killed her patient, suffers from 'survivor guilt' and 'doesn't sleep at night', a court has heard."
word = water.split()
x = 0
for words in word:
    print(words,end=" ")
    x += 1
    if x == 10:
        print()  # move to next line
        x = 0
print()
for i in range(1,11):
    print(f"{2} X {i} = {2*i}")