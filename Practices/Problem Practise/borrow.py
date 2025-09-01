class library:
    def __init__(self,name,book):
        self.name = name
        self.book = book

    def borrow(self):
        print(f"{self.name} borrow a book name {self.book}")
    
    @staticmethod
    def permisin():
        print(f"Your have to submit this book at time")
name = input("Enter your name: ")
book = input("Enter your book name: ")
z = library(name,book)
z.borrow()
z.permisin()

