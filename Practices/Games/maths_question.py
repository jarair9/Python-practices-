# import random
# import time

# operators = ["+","-","*"]
# max_num = 10
# min_num = 2
# total_num = 10

# def generators():
#     left = random.randint(min_num,max_num)
#     right = random.randint(min_num,max_num)
#     operator = random.choice(operators)

#     expr = str(left)  + ""+ operator + ""+str(right)
#     answer = eval(expr)
#     return answer, expr

# wrong = 0
# print("Press enter to start")
# print("----------------------")
# start = time.time()
# for i in range(total_num):
#     while True:
#         answer, expr = generators()
#         guess = input("Problem #"+ str(i +1) + ":" " " +  expr  + " " "=")
#         if guess == str(answer):
#             break
#         wrong += 1

# end = time.time()

# ans = round(end - start ,2)
# print("_________________")
# print(f"you have done it in {ans} seconds!")


import random
import time


OPERATORS = ["+", "-", "*"]
MIN_OPERAND = 3
MAX_OPERAND = 12
TOTAL_PROBLEMS = 10


def generate_problem():
    left = random.randint(MIN_OPERAND, MAX_OPERAND)
    right = random.randint(MIN_OPERAND, MAX_OPERAND)
    operator = random.choice(OPERATORS)

    expr = str(left) + " " + operator + " " + str(right)
    answer = eval(expr)
    return expr, answer


wrong = 0
input("Press enter to start!")
print("----------------------")

start_time = time.time()

for i in range(TOTAL_PROBLEMS):
    expr, answer = generate_problem()
    while True:
        guess = input("Problem #" + str(i + 1) + ": " + expr + " = ")
        if guess == str(answer):
            break
        wrong += 1

end_time = time.time()
total_time = round(end_time - start_time, 2)

print("----------------------")
print("Nice work! You finished in", total_time, "seconds!")
