import tqdm
import torch
import torch.nn as nn
from imdb_data import IMDBdata


class NBOW(nn.Module):
	def __init__(self, VOCAB_SIZE, DIM_EMB=300, NUM_CLASSES=2):
		super(NBOW, self).__init__()
		self.NUM_CLASSES = NUM_CLASSES

	# TODO: Initialize parameters.

	def forward(self, X):
		# TODO: Implement forward computation.
		return torch.randn(self.NUM_CLASSES)


def EvalNet(data, net):
	num_correct = 0
	Y = (data.Y + 1.0) / 2.0
	X = data.XwordList
	for i in range(len(X)):
		logProbs = net.forward(X[i])
		pred = torch.argmax(logProbs)
		if pred == Y[i]:
			num_correct += 1
	print("Accuracy: %s" % (float(num_correct) / float(len(X))))


def SavePredictions(data, outFile, net):
	fOut = open(outFile, 'w')
	for i in range(len(data.XwordList)):
		logProbs = net.forward(data.XwordList[i])
		pred = torch.argmax(logProbs)
		fOut.write(f"{data.XfileList[i]}\t{pred}\n")


def Train(net, X, Y, n_iter, dev):
	print("Start Training!")
	# TODO: initialize optimizer.

	num_classes = len(set(Y))

	for epoch in range(n_iter):
		num_correct = 0
		total_loss = 0.0
		net.train()  # Put the network into training mode
		for i in tqdm.notebook.tqdm(range(len(X))):
			pass
		# TODO: compute gradients, do parameter update, compute loss.
		net.eval()  # Switch to eval mode
		print(f"loss on epoch {epoch} = {total_loss}")
		EvalNet(dev, net)


train = IMDBdata("aclImdb_small/train")
train.vocab.Lock()
dev = IMDBdata("aclImdb_small/dev", vocab=train.vocab)
test = IMDBdata("aclImdb_small/test", vocab=train.vocab)

nbow_model = NBOW(train.vocab.GetVocabSize()).cuda()
Train(nbow_model, train.XwordList, (train.Y + 1.0) / 2.0, 5, dev)
EvalNet(test, nbow_model)
SavePredictions(test, 'test_pred_nbow.txt', nbow_model)
