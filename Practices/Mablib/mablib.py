with open("story.txt", "r", encoding="utf-8") as f:
    content = f.read()
    

words = set()
start_of_word = -1
start = ">"
end = "<"

for i , char in enumerate(content):
    if char == start:
        start_of_word = i
    elif char == end and start_of_word != -1:
        word = content[start_of_word: i +1]
          # --- DEEP EXPLANATION ---
        # story[start:i+1] means:
        # start: the index where '>' was found (for example, index 10)
        # i: the index where '<' was found (for example, index 15)
        # i+1: we go up to one past the '<' so we include it
        # story[start:i+1] → get the substring from '>' to '<' including both
        # Example: story[10:16] → '>noun<' (from index 10 to 15, because 16 is excluded)
        
        words.add(word)
        start_of_word = -1

answers = {}

for word in words:
    answer = input(f"Enter a "+ word + ": ")
    answers[word] = answer

for word in words:
    content = content.replace(word, answers[word])
print(content)





# with open("story.txt", "r", encoding="utf-8") as f:
#     content = f.read()

# placeholders = set()
# start_of_word = -1
# start = ">"
# end = "<"

# for i, char in enumerate(content):
#     if char == start:
#         start_of_word = i + 1  # Skip the ">"
#     elif char == end and start_of_word != -1:
#         word = content[start_of_word:i]  # Extract text between > and <
#         placeholders.add(word)
#         start_of_word = -1  # Reset after capture

# # Display results
# for word in placeholders:
#     print(word)


    
