import numpy as np
from datetime import datetime
# Training the AI

class Trainer():
    def __init__(self, policy, normalizer, hyperParameters, explorator):
        self.policy = policy
        self.normalizer = normalizer
        self.hyperParameters = hyperParameters
        self.explorator = explorator

    def train(self):

        for step in range(self.hyperParameters.numberOfSteps):

            print('Step:', step, 'START:', datetime.now())
            # Initializing the perturbations deltas and the positive/negative rewards
            deltas = self.policy.samplePerturbation()
            positiveRewards = [0] * self.hyperParameters.numberOfDirecitons
            negativeRewards = [0] * self.hyperParameters.numberOfDirecitons
            
            # Getting the positive rewards in the positive directions
            for k in range(self.hyperParameters.numberOfDirecitons):
                print('    Positive START:', datetime.now())
                positiveRewards[k] = self.explorator.explore(direction = True, delta = deltas[k])
                print('    Positive END:', datetime.now())
            
            # Getting the negative rewards in the negative/opposite directions
            for k in range(self.hyperParameters.numberOfDirecitons):
                print('    Negative START:', datetime.now())
                negativeRewards[k] = self.explorator.explore(direction = False, delta = deltas[k])
                print('    Negative END:', datetime.now())
            
            # Gathering all the positive/negative rewards to compute the standard deviation of these rewards
            allRewards = np.array(positiveRewards + negativeRewards)
            sigmaR = allRewards.std()
            
            # Sorting the rollouts by the max(rPositive, rNegative) and selecting the best directions
            scores = {k:max(rPositive, rNegative) for k,(rPositive,rNegative) in enumerate(zip(positiveRewards, negativeRewards))}
            order = sorted(scores.keys(), key = lambda x:scores[x], reverse = True)[:self.hyperParameters.numberOfBestDirections]
            rollouts = [(positiveRewards[k], negativeRewards[k], deltas[k]) for k in order]
            
            # Updating our policy
            self.policy.update(rollouts, sigmaR)
            
            # Printing the final reward of the policy after the update
            print('  None START:', datetime.now())
            rewardEvaluation = self.explorator.explore()
            print('  None End:', datetime.now())
            print('Step:', step, 'Reward:', rewardEvaluation, ':', datetime.now())
        print('-final policy', self.policy)
        print('-final theta', self.policy.theta)