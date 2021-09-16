import numpy as np
from imdb_data import IMDBdata


class Eval:
	def __init__(self, pred, gold):
		self.pred = pred
		self.gold = gold

	def Accuracy(self):
		return np.sum(np.equal(self.pred, self.gold)) / float(len(self.gold))


class Perceptron:
	def __init__(self, X, Y, N_ITERATIONS):
		# TODO: Initalize parameters
		self.Train(X, Y)

	def ComputeAverageParameters(self):
		# TODO: Compute average parameters (do this part last)
		return

	def Train(self, X, Y):
		# TODO: Estimate perceptron parameters
		return

	def Predict(self, X):
		# TODO: Implement perceptron classification
		return [1 for i in range(X.shape[0])]

	def SavePredictions(self, data, outFile):
		Y_pred = self.Predict(data.X)
		fOut = open(outFile, 'w')
		for i in range(len(data.XfileList)):
			fOut.write(f"{data.XfileList[i]}\t{Y_pred[i]}\n")

	def Eval(self, X_test, Y_test):
		Y_pred = self.Predict(X_test)
		ev = Eval(Y_pred, Y_test)
		return ev.Accuracy()


train = IMDBdata("aclImdb_small/train")
train.vocab.Lock()
print("Train")
dev = IMDBdata("aclImdb_small/dev", vocab=train.vocab)
test = IMDBdata("aclImdb_small/test", vocab=train.vocab)
print("Test")

ptron = Perceptron(train.X, train.Y, 10)
ptron.ComputeAverageParameters()

print(ptron.Eval(test.X, test.Y))
ptron.SavePredictions(test, 'test_pred_perceptron.txt')

# TODO: Print the 20 most positive and 20 most negative words
