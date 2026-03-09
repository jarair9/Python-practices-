from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

app = FastAPI(
    title="Book Store API",
    description="A simple example API with FastAPI",
    version="1.0.0"
)

# Define a data model using Pydantic
class Book(BaseModel):
    id: int
    title: str
    author: str
    price: float
    description: Optional[str] = None

# In-memory database
books_db = []

# Home route
@app.get("/")
def home():
    return {"message": "Welcome to the Book Store API!"}

# List all books
@app.get("/books")
def get_books():
    return {"books": books_db}

# Get a single book by ID
@app.get("/books/{book_id}")
def get_book(book_id: int):
    for book in books_db:
        if book["id"] == book_id:
            return book
    return {"error": "Book not found"}

# Add a new book
@app.post("/books")
def create_book(book: Book):
    books_db.append(book.dict())
    return {"message": "Book added successfully", "book": book}

# Search books by author (query param)
@app.get("/search")
def search_books(author: Optional[str] = None):
    if not author:
        return {"error": "Please provide an author to search"}
    results = [book for book in books_db if book["author"].lower() == author.lower()]
    return {"results": results}
