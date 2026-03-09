import time 
import winsound



def alarmclk():
    i = 0
   
    while True:
        min_re=  i // 60
        sec_re =  i % 60
        print(f"{min_re} : {sec_re:02d}",end="\r",flush=True)
        time.sleep(0.1)
        i +=1
        if min_re == target_min and sec_re == target_sec:
            print("Times up")
            break
        

        # if min_re or sec_re == min or sec:
        #     winsound.Beep(1000,2000)
target_min = int(input("Enter a minute: "))
target_sec = int(input("Enter a seconds: "))
alarmclk()

# import time
# # import winsound  # Only works on Windows

# def alarmclk(minutes, seconds):
#     total_seconds = minutes * 60 + seconds

#     while total_seconds >= 0:
#         min_re = total_seconds // 60
#         sec_re = total_seconds % 60
#         print(f"{min_re:02d}:{sec_re:02d}", end="\r", flush=True)
#         time.sleep(1)
#         total_seconds -= 1

#     print("\nTime's up!")
   

# # Get user input
# minutes = int(input("Enter minutes: "))
# seconds = int(input("Enter seconds: "))
# alarmclk(minutes, seconds)


    