import torch
import torch.nn as nn
from imdb_data import IMDBdata
from utils import Train, EvalNet, SavePredictions

class NBOW(nn.Module):
	def __init__(self, VOCAB_SIZE, DIM_EMB=300, NUM_CLASSES=2):
		super(NBOW, self).__init__()
		self.NUM_CLASSES = NUM_CLASSES

	# TODO: Initialize parameters.

	def forward(self, X):
		# TODO: Implement forward computation.
		return torch.randn(self.NUM_CLASSES)


train = IMDBdata("aclImdb_small/train")
train.vocab.Lock()
dev = IMDBdata("aclImdb_small/dev", vocab=train.vocab)
test = IMDBdata("aclImdb_small/test", vocab=train.vocab)

nbow_model = NBOW(train.vocab.GetVocabSize()).cuda()
Train(nbow_model, train.XwordList, (train.Y + 1.0) / 2.0, 5, dev)
EvalNet(test, nbow_model)
SavePredictions(test, 'test_pred_nbow.txt', nbow_model)
