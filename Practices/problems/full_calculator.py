# def squareroot(n1,n2):
#     result = 
#     return result
# def cuberoot(n1):
#     result = 
#     return result

def add(n1,n2):
    result = n1 + n2
    print(result)
def sub(n1,n2):
    result = n1 - n2
    print(result)
def mul(n1,n2):
    result = n1 * n2
    print(result)
def div(n1,n2):
    result = n1 / n2
    print(result)
def sqaure(n1):
    result = n1 ** 2
    print(result)
    
def main():
    print("Welcome to be here")
    print("This is the python calculator \n")
    print("q for quit")
    while True:
        print("\nSelect an operators")
        print("1. Add")
        print("2. sub")
        print("3. mul")
        print("4. div")
        print("5. sqaure")
        # print("6. square root")
        # print("7. cube root")
        operator = input("Enter an operator: ")
        if operator == 1 or operator == "Add".lower():
            n1 = int(input("Enter a number : "))
            n2 = int(input("Enter a number : "))
            add(n1,n2)
        elif operator == 2 or operator == "Sub".lower():
            n1 = int(input("Enter a number : "))
            n2 = int(input("Enter a number : "))
            sub(n1,n2)
        elif operator == 3 or operator == "mul".lower():
            n1 = int(input("Enter a number : "))
            n2 = int(input("Enter a number : "))
            mul(n1,n2)
        elif operator == 4 or operator == "div".lower():
            n1 = int(input("Enter a number : "))
            n2 = int(input("Enter a number : "))
            div(n1,n2)
        elif operator == 5 or operator == "square".lower():
            n1 = int(input("Enter a number : "))
            sqaure(n1)
        elif operator == "q" or operator == "Q":
            break
            
       
main()        

