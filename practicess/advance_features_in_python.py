# map()
# filter()
# reduce()
# zip()
# enumerate()

from functools import reduce
def square(x):
    return x * x
def is_even(x):
    return x % 2 == 0
def add(x, y):
    return x + y
def main():
    numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    # Using map to square each number
    squared_numbers = list(map(square, numbers))
    print("Squared Numbers:", squared_numbers)
    
    # Using filter to get even numbers
    even_numbers = list(filter(is_even, numbers))
    print("Even Numbers:", even_numbers)
    
    # Using reduce to sum all numbers
    total_sum = reduce(add, numbers)
    print("Total Sum:", total_sum)
    
    # Using zip to pair elements from two lists
    letters = ['a', 'b', 'c', 'd', 'e']
    paired = list(zip(numbers, letters))
    print("Paired Numbers and Letters:", paired)

    # Using enumerate to get index and value
    for index, value in enumerate(numbers, start=1):
        print(f"Index: {index}, Value: {value}")

main()

"OR"

numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
squared_numbers = list(map(lambda x: x * x, numbers))

print("Squared Numbers:", squared_numbers)
even_numbers = list(filter(lambda x: x % 2 == 0, numbers))
print("Even Numbers:", even_numbers)

from functools import reduce

total_sum = reduce(lambda x, y: x + y, numbers)
print("Total Sum:", total_sum)

letters = ['a', 'b', 'c', 'd', 'e']
paired = list(zip(numbers, letters))
print("Paired Numbers and Letters:", paired)

# for index, value in enumerate(numbers, start=1):
#     print(f"Index: {index}, Value: {value}")
