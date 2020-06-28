import numpy

# Normalization

# States: Vectors representing what is happening in the environment

class Normalizer():

    # self: he instantiation, numberoOfInputs: number of imputs (elements of vector) of the perceptron (neuron)
    def __init__(self, numberoOfInputs): 
        self.counter = numpy.zeros(numberoOfInputs)
        self.mean = numpy.zeros(numberoOfInputs)
        self.meanDiff = numpy.zeros(numberoOfInputs) # Numerator
        self.variant = numpy.zeros(numberoOfInputs)

    def observe(self, newState):
        self.counter += 1. # All the values of the vector will be incremented by 1.0
        # mean computation
        previousMean =  self.mean.copy()
        self.mean += (newState - self.mean) / self.counter # Computation online
        # Variant computation
        self.meanDiff += (newState - previousMean) * (newState - self.mean)
        self.variant = (self.meanDiff / self.counter).clip(min = 1e-2)

    def normalize(self, inputs):
        observedMean = self.mean
        observedStandarDerivation = numpy.sqrt(self.variant)
        return (inputs - observedMean) / observedStandarDerivation
