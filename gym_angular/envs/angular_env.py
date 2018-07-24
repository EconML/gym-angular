import gym
from gym import error, spaces, utils
from gym.uitls import seeding
import sys, os
import numpy as np
import cv2
import glob

class AngularEnv(gym.Env):

    #Possibly rework/relabel with more inc/dec options
    ACTION_SPACE={'increase':0.01,'decrease':-0.01,'no_change':0}

    CASP_PATH = 'd:/ML/Code/img_process/T0689/'


    metadata = {'render.modes': ['human']}

    def __init__(self):
        #episode prediction incremental tally
        self.prediction_total = 0.0
        #create numpy array containing the model as a sequence of images
        self.imgs = self.load_imgs(CASP_PATH)

        self.action_space = spaces.Discrete(len(ACTION_SPACE))
        self.observation_space = spaces.Box(
            low = 0,
            high = 255,
            shape = (WINDOW_SIZE, NUM_FEATURES, 4),
            dtype = np.float64
        )
        self.done = False

        #render counter
        self.render_count = 0

        #time_steps for
        self.time_steps = 1



    #one time step in the environments dynamics
    def step(self, action):
        #amend prediction to prediction_total
        self.prediction_total += ACTION_SPACE[action]

        if self.done:
            return

        #get next set of images or possibly one image -> handle stack in RL method
        next_state = self.imgs[self.time_steps]
        self.time_steps+=1

        #if done with model calculate episode reward as between prediction_total and GDT
        if self.time_steps == len(self.imgs):
            self.done = True
            reward = math.abs(GDT-self.prediction_total) * -1
            return next_state, reward, self.done, {}

        #step reward
        reward = +0.01
        return next_state, reward, self.done, {}


    """
    Resets the state of the environment and returns intitial observation

    args: path - the path for the next model/target file

    Returns: observation(object) the initial obs of the new episode
    """
    def reset(self, path):

        #Reset prediciton total for next model
        self.prediction_total = 0.0
        #Call to load_imgs to create the sequence of images
        self.imgs = self.load_imgs(path)
        #return first element in model img file
        self.time_steps = 1
        return self.imgs[0]

    """"
    return formatted string of current location within model image set
    """
    def render(self, mode='human',model):
        print('Working on Target: ' + target_path +
            'Model: ' + model + 'Image: ' + render_count)
        render_count+=1

    """
    returns image for a given path
    """
    def process_state(self, path):
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return image


    """
    Creates numpy array containing all images for a given model

    args: file - a target file
          model - the models name

    returns: a numpy array of images in shape (num_images,WINDOW_SIZE,NUM_FEATURES)
    """
    def load_imgs(target_path, model):
        #instantiate empty array to hold images
        img_data_temp = []
        #load each png image into the temp array
        for img_file in glob.glob(target_path + model + '/*.png):
            image = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
            img_data_temp.append(image)
        img_data = np.array(img_data_temp)
        return img_data
