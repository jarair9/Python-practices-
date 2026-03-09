# decorators in Python
def my_decorator(func):
    def king():
        print("Before function")
        func() # this call means that we can call another function
        print("After function")
    return king
    

@my_decorator
def say_hello():
    print("Hello!")

say_hello()

# # decorators with arguments
def repeat(func):
    def wrapper(*args, **kwargs):
        func(*args, **kwargs)
        func(*args, **kwargs)
    return wrapper

@repeat
def greet(name):
    print(f"Hello, {name}!")

greet("Alice")


# decorators with parameters
class MyClass:
    @staticmethod
    def hello():
        print(3*5)

MyClass.hello()

# decorators with properties
class Person:
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name

p = Person("Bob")
print(p.name)

def loop(func):
    def wrapper():
        func()
    return wrapper

@loop
def looop():
    num = ["a","b","c","d","e"]
    
