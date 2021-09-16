import torch
import torch.nn as nn
from utils import Train, EvalNet, SavePredictions
from imdb_data import IMDBdata


class CNN(nn.Module):
	def __init__(self, VOCAB_SIZE, DIM_EMB=300, NUM_CLASSES=2):
		super(CNN, self).__init__()
		self.NUM_CLASSES = NUM_CLASSES

	# TODO: Initialize parameters.

	def forward(self, X):
		# TODO: Implement forward computation.
		return torch.randn(self.NUM_CLASSES)


train = IMDBdata("aclImdb_small/train")
train.vocab.Lock()
dev = IMDBdata("aclImdb_small/dev", vocab=train.vocab)
test = IMDBdata("aclImdb_small/test", vocab=train.vocab)

cnn = CNN(train.vocab.GetVocabSize()).cuda()
Train(cnn, train.XwordList, (train.Y + 1.0) / 2.0, 5, dev)

EvalNet(test, cnn)
SavePredictions(test, 'test_pred_cnn.txt', cnn)
