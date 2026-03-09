from pathlib import Path

# Access a directory
dir_path = Path("C:/Users/Shaheen Alam/Desktop/python practices")

# Check if it exists
if dir_path.exists():
    print("Directory exists")

# List files
for item in dir_path.iterdir():
    print(item)




# finding file in directory 
from pathlib import Path

dir_path = Path("C:/Users/Shaheen Alam/Desktop/python practices")
file_path = dir_path / "program10.py"

if file_path.exists():
    print("File found:", file_path.name)
else:
    print("File not found.")



for item in dir_path.iterdir():
    if item.is_file():
        print(item.name)
