#!/usr/bin/env python

#####################################################################
# This script presents how to run some scenarios.
# Configuration is loaded from "../../scenarios/<SCENARIO_NAME>.cfg" file.
# <episodes> number of episodes are played. 
# Random combination of buttons is chosen for every action.
# Game variables from state and last reward are printed.
#
# To see the scenario description go to "../../scenarios/README.md"
#####################################################################

from __future__ import print_function

import numpy as np
import itertools as it
from random import choice
from time import sleep
from vizdoom import DoomGame, ScreenResolution, ScreenFormat
import cv2
import skimage.color, skimage.transform

game = DoomGame()

# Choose scenario config file you wish to watch.
# Don't load two configs cause the second will overrite the first one.
# Multiple config files are ok but combining these ones doesn't make much sense.

# game.load_config("../../scenarios/basic.cfg")
# game.load_config("../../scenarios/simpler_basic.cfg")
# game.load_config("../../scenarios/rocket_basic.cfg")
# game.load_config("../../scenarios/deadly_corridor.cfg")
# game.load_config("../../scenarios/deathmatch.cfg")
# game.load_config("../../scenarios/defend_the_center.cfg")

game.load_config("../../scenarios/defend_the_line.cfg")
# game.load_config("../../scenarios/health_gathering.cfg")
# game.load_config("../../scenarios/my_way_home.cfg")
# game.load_config("../../scenarios/predict_position.cfg")
# game.load_config("../../scenarios/take_cover.cfg")

game.set_screen_format(ScreenFormat.RGB24)
# Makes the screen bigger to see more details.
game.set_screen_resolution(ScreenResolution.RES_640X480)
game.set_window_visible(False)
game.set_labels_buffer_enabled(True)
game.set_depth_buffer_enabled(True)
game.init()

resolution = (200, 350)

img0 = None
img1 = None
img2 = None
img3 = None

# Creates all possible actions depending on how many buttons there are.
actions_num = game.get_available_buttons_size()
actions = []
for perm in it.product([False, True], repeat=actions_num):
    actions.append(list(perm))

episodes = 10
sleep_time = 0.028

for i in range(episodes):
    print("Episode #" + str(i + 1))

    # Not needed for the first episode but the loop is nicer.
    currentFrameIndex = 0
    currentImgIndex = 0
    imagesFilled = False
    fourFrames = [None, None, None, None]
    game.new_episode()
    while not game.is_episode_finished():

        # Gets the state and possibly to something with it
        state = game.get_state()

        # Makes a random action and save the reward.
        reward = game.make_action(choice(actions))

        img = state.screen_buffer
        
        #depthAndMask = cv2.addWeighted(mask,1.0,depth,4.0,0)      
        ##masked = cv2.bitwise_and(img, img, mask=mask)

        #edges = cv2.Canny(mask, 100, 150)

        #cv2.imshow('ViZDoom Label Buffer', img)
        ##cv2.imshow('Masked', masked)
        ##cv2.imshow('Labels', depthAndMask)
        ##cv2.imshow('Edges', mask)
        ##cv2.waitKey(1)

        print("currentFrameIndex", currentFrameIndex)
        if((currentFrameIndex + 1) % 5 == 0):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = skimage.transform.resize(img, resolution)
            img = img.astype(np.float32)
            
            print("assigned image index ", currentImgIndex)
            cv2.putText(img,str(currentFrameIndex), (25,25), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
            img3 = img2
            img2 = img1
            img1 = img0
            img0 = img

        if(type(img3) != type(None)):
            print("showing images")
            cv2.imshow('Frame 1', img0)
            cv2.imshow('Frame 2', img1)
            cv2.imshow('Frame 3', img2)
            cv2.imshow('Frame 4', img3)
            cv2.waitKey(1)

        currentFrameIndex = currentFrameIndex +1
        
        #print("State #" + str(state.number))
        #print("Game Variables:", state.game_variables)
        #print("Performed action:", game.get_last_action())
        #print("Last Reward:", reward)
        print("=====================")

        # Sleep some time because processing is too fast to watch.
        if sleep_time > 0:
            sleep(sleep_time)


    print("Episode finished!")
    print("total reward:", game.get_total_reward())
    print("************************")

cv2.destroyAllWindows()