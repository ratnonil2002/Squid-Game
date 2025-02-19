import pygame
import time

def playAudio(audioFilePath='./squid.mp3'):
    pygame.init()

    pygame.mixer.music.load(audioFilePath)

    pygame.mixer.music.play()

    # while pygame.mixer.music.get_busy():
    #     pygame.time.Clock().tick(10)

global gameOn

def loopAudio():
    time.sleep(3)
    global gameOn
    while gameOn:
        playAudio()
        breakFlag = False
        for i in range(50):
            if not gameOn:
                breakFlag = True
                break
            else:
                time.sleep(0.2)
        if breakFlag:
            break
        time.sleep(2)