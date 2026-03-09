# Making an library that contains book and a student borrow a book so the student name and book are print in result


class student:
    def __init__(self,name,book):
        self.name = name
        self.book = book


    def borrow(self):
        print(f"{self.name} borrow a book  name {self.book}")


name = str(input("Enter your name : "))
book = str(input("Enter your book name or title: "))


a = student(name,book)
a.borrow()