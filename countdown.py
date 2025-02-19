import time 

global currTime
global timeOver
currTime = ""
timeOver = False
MINUTES = 1

def countTime():
    global currTime
    global timeOver
    currTime = "0" + str(MINUTES) + ":00"
    time.sleep(5)
    
    for minute in range(MINUTES - 1, -1, -1):
        for second in range(59, -1, -1):
            minuteStr = str(minute)
            secondStr = str(second)
            if len(minuteStr) == 1:
                minuteStr = "0" + minuteStr
            if len(secondStr) == 1:
                secondStr = "0" + secondStr
            currTime = minuteStr + ":" + secondStr
            time.sleep(1)

    timeOver = True