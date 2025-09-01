# # Importing the Counter class from the collections module to count occurrences of elements
# from collections import Counter     
# # Defining the main function to process a paragraph
# def paragraph():
#     # Taking input from the user as a string
#     prph = str(input("Enter a paraghraph : "))
#     # Iterating through each character in the input paragraph
#     for words in prph:
#         # Checking if the character exists in the paragraph (this condition is redundant)
#         if words in prph:
#             # Attempting to count occurrences of the character (but the count function is not defined)
#             counter = counter(words)
#             # Printing the count of the character
#             print(counter)

# # Calling the paragraph function to execute the code
# paragraph()



########AI#########

# Importing the Counter class from the collections module to count occurrences of elements
from collections import Counter

# Defining the main function to process a paragraph
def paragraph():
    # Taking input from the user as a string
    prph = str(input("Enter a paragraph: "))
    
    # Splitting the paragraph into words
    words = prph.split()
    
    # Counting the occurrences of each word using Counter
    counter = Counter(words)
    
    # Printing the word frequency
    print("Word Frequency:")
    for word, count in counter.items():
        print(f"{word}: {count}")

# Calling the paragraph function to execute the code
paragraph()






