import time
import audio

global currLight 
global gameOn

# gameOn = False
currLight = (0, 0, 255)

def loopLight(): 
    global gameOn  # Declare gameOn as global within loopLight function
    time.sleep(3)
    global currLight

    while gameOn:
        if currLight == (0, 0, 255):
            currLight = (0, 255, 0)
            time.sleep(8)
        else:
            currLight = (0, 0, 255)
            time.sleep(4)