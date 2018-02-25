# 创建感知
import numpy as np

class Perceptron:
	"""
	This class models an artificial neuron with step activation function.
	"""
	def __init__(self, weights=np.array([1]), threshold=0):
		"""
		initialize weights and threshold based on input arguments.
		"""
		self.weights = weights
		self.threshold = threshold

	def activate(self, inputs):
		"""	
		Takes in @params inputs, a list of numbers equal to length of weights.
		@return the output of a threshold perceptron with given inputs based on 
		perceptron weights and threshold.
		"""
		strength = np.dot(self.weights, inputs)
		# if strength <= self.threshold:
		# 	self.result = 0
		# else:
		# 	self.result = 1
		# return self.result
		return int(strength > self.threshold)

	def update(self, values, train, eta=.1):
		"""
		Takes in a 2D array @param values consisting of a LIST of inputs and a
		1D array @param train, consisting of a corresponding list of expected
		outputs. Updates internal weights according to the perceptron training 
		rule using these values and an optional learning rate, @param eta.
		"""

def test():
	p1 = Perceptron(np.array([1, 2]), 0.)
	assert p1.activate(np.array([ 1,-1])) == 0 # < threshold --> 0
	assert p1.activate(np.array([-1, 1])) == 1 # > threshold --> 1
	assert p1.activate(np.array([ 2,-1])) == 0 # on threshold --> 0

if __name__ == "__main__":
	test()













