#adding task to a list
#creating a list
#user can add task to the list
#user can remove task from the list
#user can view the list of tasks

#creating a list
to_do_list = []
def add_task():
    print("Welcome to to do list manager app")
    print("if want to remove task from the list enter 'remove' ")
    print("_"*10)
    while True:
        task = input("Enter a task to add in the list : ")
        if task == "remove":
            print(f"The task are {to_do_list}")
            removed = input("\nEnter the task to remove: ")
            if removed not in to_do_list:
                print("your task is not list")
            else:
                to_do_list.remove(removed)
                print(f"your {removed} task is removed from a list")
            
        
        to_do_list.append(task)
        print(f"Your task has been added to a list ")
        
add_task()


   