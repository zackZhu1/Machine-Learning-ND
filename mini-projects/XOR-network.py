import numpy as np

class Perceptron:
    """
    This class models an artificial neuron with step activation function.
    """
    def __init__(self, weights = np.array([1]), threshold = 0):
        """
        :param weights:
        :param threshold:
        """
        self.weights = weights
        self.threshold = threshold

    def activate(self, values):
        """
        :param values: a list of numbers equal to length of weights
        :return: the output of a threshold perceptron with given inputs based on perceptron weights and threshold.
        """
        # First calculate the strength with which the perceptron fires
        strength = np.dot(values, self.weights)
        # Then return 0 or 1 depending on strength compared to threshold
        return int(strength > self.threshold)


# Part 1: Set up the perceptron network
Network = [
    # input layer
    [Perceptron(np.array([1, -1]), 0), Perceptron(np.array([-1, 1]), 0)],
    # output layer
    [Perceptron(np.array([1, 1]), 0)]
]

# Part2: Define a procedure to compute the output of the network
def EvalNetwork(inputValues, Network):
    """
    :param inputValues:
    :param Network:
    :return: the output value is a single number
    """
    OutputValue = inputValues
    for layer in Network:
        OutputValue = map(lambda p : p.activate(OutputValue), layer)
    return OutputValue[0]

def test():
    print "0 XOR 0 = 0?:", EvalNetwork(np.array([0, 0]), Network)
    print "0 XOR 1 = 1?:", EvalNetwork(np.array([0, 1]), Network)
    print "1 XOR 0 = 1?:", EvalNetwork(np.array([1, 0]), Network)
    print "1 XOR 1 = 0?:", EvalNetwork(np.array([1, 1]), Network)

if __name__ == "__main__":
    test()