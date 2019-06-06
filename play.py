#! /usr/bin/env python
from io import StringIO

import numpy as np
from pygame.locals import KEYDOWN, K_ESCAPE, K_x, QUIT
from PIL import Image
import pygame
import rpg.states
from controler import DQNAgent
"""
If using an older Raspberry Pi distro, you might need to run the following commands to get working sound:

sudo apt-get install alsa-utils
sudo modprobe snd_bcm2835
"""

# initialise pygame before we import anything else
pygame.mixer.pre_init(44100, -16, 2, 1024)
pygame.init()
state_size = (160, 160, 4)
action_size = 5  # UP, DOWN, LEFT, RIGHT, SPACE
agent = DQNAgent(state_size, action_size)
done = False
batch_size = 32


def playMain():
    # get the first state
    currentState = rpg.states.showTitle(True)
    # start the main loop
    clock = pygame.time.Clock()    
    while True:
        clock.tick(rpg.states.FRAMES_PER_SEC)
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                return
            if event.type == KEYDOWN and event.key == K_x:
                # toggle sound
                # rpg.states.soundHandler.toggleSound()
                # rpg.states.musicPlayer.toggleMusic()
                pass

        # detect key presses    
        keyPresses = [0]*323
        # if max(keyPresses) != 0:
        #     for index, item in enumerate(keyPresses):
        #         print(index)
        # pygame.image.save(currentState.backgroundImage, data)
        data = pygame.image.tostring(rpg.states.screen, 'RGB')
        img = Image.frombytes('RGB', (160, 160), data)
        observable = np.asarray(img).astype('float64')
        observable /= 255.
        action = agent.act(observable)
        if action == 0:
            keyPresses[pygame.K_LEFT] = 1
        elif action == 1:
            keyPresses[pygame.K_RIGHT] = 1
        elif action == 2:
            keyPresses[pygame.K_UP] = 1
        elif action == 3:
            keyPresses[pygame.K_DOWN] = 1
        else:
            keyPresses[pygame.K_SPACE] = 1

        # delegate key presses to the current state
        newState = currentState.execute(keyPresses)
        # print(currentState.playState)
        reward = -1
        # agent.remember(currentState, action, reward, newState, done)
        # flush sounds
        rpg.states.soundHandler.flush()
        # change state if necessary
        if newState:
            currentState = newState


# this calls the playMain function when this script is executed
if __name__ == '__main__':
    playMain()
