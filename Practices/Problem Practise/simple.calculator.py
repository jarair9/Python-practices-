# calculator

def add(x,y):
    return x+y

def sub(x,y):
    return x-y

def mul(x,y):
    return x*y

def div(x,y):
    return x/y

print("Select operation:")
print("add")
print("sub")
print("mul")
print("div")
operator = str(input("Enter an operator: ")).lower()
x = int(input("Enter a number: "))
y = int(input("Enter a number: "))

if operator == "add":
    print(add(x , y))
if operator == "sub":
    print(sub(x , y))
if operator == "mul":
    print(mul(x , y))
if operator == "div":
    print(div(x , y))


