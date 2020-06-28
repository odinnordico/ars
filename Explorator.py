# Exploration
# env: the environment where the policy(ai) will be
# normalizer: normalize the states/inputs
# policy: ai
# direction:
# delta: perturbation

class Explorator():

    def __init__(self, hyperParameters, normalizer, policy, enviroment):
        self.hyperParameters = hyperParameters
        self.enviroment = enviroment
        self.normalizer = normalizer
        self.policy = policy
        
    def explore(self, direction = None, delta = None):
        state = self.enviroment.reset()
        numberOfActionPlayed = 0.
        rewartSum = 0
        done = False
        while not done and numberOfActionPlayed < self.hyperParameters.episodeLength:
            # feed the perceptron of the policy
            self.normalizer.observe(state)
            state = self.normalizer.normalize(state)
            action = self.policy.evaluate(state, delta, direction)
            state, reward, done, unused = self.enviroment.step(action)
            reward = max(min(reward, 1), -1)
            rewartSum += reward
            numberOfActionPlayed += 1
        return rewartSum
