import numpy
import pybullet_envs
# Hyper parameters
# Hyper parameter: Parameter suposed to be a fixed parameter, can be any variable or other kind of variable.

class HyperParameter():

    def __init__(self):
        self.numberOfSteps = 1000 # Number of time to update the model
        self.episodeLength = 1000 # Maximun length of an episode
        self.learningRate = 0.02 # How fast the AI is learning
        self.numberOfDirecitons = 16 #Number of perturbations to be applied; the more directions are defined the more time to train the model will take
        self.numberOfBestDirections = 16 #must be lower than numberOfDirections
        self.noise = 0.03 # Signal in the gaussian distribution to sample the perturbations
        self.seed = 1 # Fix the parameters of the environment
        self.environmentName = 'HalfCheetahBulletEnv-v0' # The environment

        assert self.numberOfBestDirections <= self.numberOfDirecitons

