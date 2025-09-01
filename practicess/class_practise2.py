class calculator:
    def __init__(self,x , y):

        self.x = x
        self.y = y

    def __mul__(self):
        return self.x*self.y

    def __add__(self):
        return self.x+self.y
    def __div__(self):
        return self.x/self.y
    def __sub__(self):
        return self.x-self.y
x = int(input("Enter x number: "))
y = int(input("Enter y number: "))

z = calculator(x,y)

print(f"Addition : {z.__add__()}")
print(f"Subtraction : {z.__sub__()}")
print(f"Multiplication: { z.__mul__()}")
print(f"division : { z.__div__()}")


