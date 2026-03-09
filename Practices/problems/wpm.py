import curses
from curses import wrapper
import time
import random
# this display the wpm and color to text that user enter
def display(stdscr, target_text, current_text, wpm):
    stdscr.addstr(0, 0, target_text)
    stdscr.addstr(1, 0, f"WPM: {wpm}")
    for i, char in enumerate(current_text): #it take index as well as value 
              correct_char = target_text[i] # it check every character in target text
              # if the char is equal to the target so it appears green color on the curent_text
              if char == correct_char: 
                  color = curses.color_pair(1)
               # the color is red 
              else: 
                  color = curses.color_pair(2)
               # it takes the user curser overlay on target text
              stdscr.addstr(0,i, char, color)
# random text taker
def loadtext():
     with open ("text.txt","r") as f:
          lines = f.readlines()
          return random.choice(lines).strip().lower()

    # this is the main program 
def wpm_test(stdscr):
     target_text = loadtext()
     current_text = []
     start_time = time.time()
     stdscr.nodelay(True)
     while True:
          stdscr.clear()
          time_elapsed = max(time.time() - start_time, 1)
          wpm = round((len(current_text) / 10) / (time_elapsed / 60))
          stdscr.addstr(0,0,target_text)
          display(stdscr, target_text, current_text, wpm)
          stdscr.refresh()
          
          if "".join(current_text) == target_text:
               stdscr.nodelay(False)
               stdscr.addstr(3, 0, "Completed! Press any key to exit.")
               stdscr.getkey()
               break
          try:
               key = stdscr.getkey()
          except:
               continue
          stdscr.refresh()

          if key in ("KEY_BACKSPACE", '\b', "\x7f"):
               if len(current_text) > 0:
                    current_text.pop()
          elif len(target_text)> len(current_text):      
            current_text.append(key)
          stdscr.refresh()

def start_screen(stdscr):
     stdscr.clear()
     stdscr.addstr("Hello this is typing speed test program")
     stdscr.addstr("\nPress any key to begin ")
     stdscr.refresh()
     stdscr.getkey()
     
def main(stdscr):
    curses.init_pair(1, curses.COLOR_GREEN,curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_RED,curses.COLOR_BLACK)
    start_screen(stdscr)
    wpm_test(stdscr)
      
wrapper(main)


