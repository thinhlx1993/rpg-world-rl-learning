#! /usr/bin/env python
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
batch_size = 32


def capture_screen():
    data = pygame.image.tostring(rpg.states.screen, 'RGBA')
    img = Image.frombytes('RGBA', (160, 160), data)
    observable = np.asarray(img).astype('float64')
    observable /= 255.
    observable = np.expand_dims(observable, axis=0)
    return observable


def playMain():
    # get the first state
    current_state = rpg.states.showTitle(True)
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
        keyPresses = pygame.key.get_pressed()
        observable = capture_screen()
        action = agent.act(observable)

        # keyPresses = [0]*323
        # if action == 0:
        #     keyPresses[pygame.K_LEFT] = 1
        # elif action == 1:
        #     keyPresses[pygame.K_RIGHT] = 1
        # elif action == 2:
        #     keyPresses[pygame.K_UP] = 1
        # elif action == 3:
        #     keyPresses[pygame.K_DOWN] = 1
        # else:
        #     keyPresses[pygame.K_SPACE] = 1

        # delegate key presses to the current state
        new_state = current_state.execute(keyPresses)
        if hasattr(current_state, 'lifeLostEvent'):
            if 'NoneType' not in str(type(current_state.lifeLostEvent)):
                if 'lives' in current_state.lifeLostEvent:
                    print(current_state.lifeLostEvent)

        reward = -1
        agent.remember(observable, action, reward, new_state, False)
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

        # flush sounds
        rpg.states.soundHandler.flush()
        # change state if necessary
        if new_state:
            current_state = new_state


# this calls the playMain function when this script is executed
if __name__ == '__main__':
    playMain()
