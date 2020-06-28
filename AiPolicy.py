# AI: Represented a a POLICY
# Explore some policies an dconvege to one

import numpy
#import ./hyperPArameter

# This is the perceptron, meaning the layer of AI wuth n neurons

# Output size can be the number of muscles and angules
class AIPolicy():
    def __init__(self, inputSize, outputSize, hyperParameters):
        # Matrice or Numbers of ways
        self.theta = numpy.zeros((outputSize, inputSize)) # Multiplication by the right
        self.hyperParameters = hyperParameters

    # Perturbation of the weights
    # delta: is the perturbation of the weight
    # direction: Positive, negative (oppsite), None
    # return: 1) Output when only input comes
    #         2) Output when a positive perturbation (delta) is provided
    #         3) Output when a perturbation (delta) is provided with positive or negative direciton
    def evaluate(self, input, delta = None, positiveDirection = None): 
        if positiveDirection is None: #Output with not perturbations applied
            return self.theta.dot(input) # dot = multiply matrix with the inputs
        elif positiveDirection is True:
            return (self.theta + self.hyperParameters.noise * delta).dot(input)
        else:
            return (self.theta - self.hyperParameters.noise * delta).dot(input)

    def samplePerturbation(self):
        # Matrix of random distributions
        # *self.theta.shape: the dimentions of the theta matrix
        return [numpy.random.rand(*self.theta.shape) for item in range(self.hyperParameters.numberOfDirecitons)]
    
    # rollout list of several triplets
    #   1) Reward of positive direction
    #   2) Reward of opposite direction
    #   3) Perturbation given by positive and opposite dirrections
    # sigmaReward: Standar deviation of the reward
    def update(self, rollout, sigmaReward):
        step = numpy.zeros(self.theta.shape)
        for positiveDirectionRewards, oppositeDirectionReward, perturbationDirrection in rollout:
            step += (positiveDirectionRewards - oppositeDirectionReward * perturbationDirrection)
        self.theta += self.hyperParameters.learningRate / (self.hyperParameters.numberOfBestDirections * sigmaReward)
