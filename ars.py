# AI 2018

# Importing the libraries
import os
import numpy as np
import gym
from gym import wrappers
import pybullet_envs

from HyperParameter import HyperParameter
from Normalizer import Normalizer
from AiPolicy import AIPolicy
from Explorator import Explorator
from Trainer import Trainer


# Running the main code

def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path

workDir = mkdir('exp', 'brs')
monitorDir = mkdir(workDir, 'monitor')

hyperParameter = HyperParameter()
np.random.seed(hyperParameter.seed)
environment = gym.make(hyperParameter.environmentName)
environment = wrappers.Monitor(environment, monitorDir, force = True)
inputsNumber = environment.observation_space.shape[0]
outputsNumber = environment.action_space.shape[0]
policy = AIPolicy(inputsNumber, outputsNumber, hyperParameter)
normalizer = Normalizer(inputsNumber)
explorator = Explorator(hyperParameter, normalizer, policy, environment)
trainer = Trainer(policy, normalizer, hyperParameter, explorator)
trainer.train()
