#####################################################################################
#TODO: Add imports if needed:

import torch.nn.functional as F
#####################################################################################


import subprocess
import torch.nn as nn
from preprocess import *
from read_data import *
from random import sample
from basic_lstm import *


class CharLSTMtagger(BasicLSTMtagger):
	def __init__(self, DIM_EMB=10, DIM_CHAR_EMB=30, DIM_HID=10, tag2i=None):
		super(CharLSTMtagger, self).__init__(DIM_EMB=DIM_EMB, DIM_HID=DIM_HID, tag2i=tag2i)
		NUM_TAGS = max(tag2i.values())+1

		(self.DIM_EMB, self.NUM_TAGS) = (DIM_EMB, NUM_TAGS)
	#####################################################################################
	#TODO: Initialize parameters.

	#####################################################################################

	def forward(self, X, X_mask, X_char, train=False):
		#####################################################################################
		#TODO: Implement the forward computation.

		return torch.randn((X.shape[0], X.shape[1], self.NUM_TAGS))  #Random baseline.
		#####################################################################################

	def sentences2input_tensors(self, sentences):
		(X, X_mask)   = prepare_input(sentences2indices(sentences, word2i))
		X_char        = prepare_input_char(sentences2indicesChar(sentences, char2i))
		return (X, X_mask, X_char)

	def inference(self, sentences):
		(X, X_mask, X_char) = self.sentences2input_tensors(sentences)
		pred = self.forward(X, X_mask, X_char).argmax(dim=2)
		return [[i2tag[pred[i,j].item()] for j in range(len(sentences[i]))] for i in range(len(sentences))]

	def print_predictions(self, words, tags):
		Y_pred = self.inference(words)
		for i in range(len(words)):
			print("----------------------------")
			print(" ".join([f"{words[i][j]}/{Y_pred[i][j]}/{tags[i][j]}" for j in range(len(words[i]))]))
			print("Predicted:\t", Y_pred[i])
			print("Gold:\t\t", tags[i])


# Training LSTM w/ character embeddings.

def train_char_lstm(sentences, tags, lstm):
	#####################################################################################
	# TODO: initialize optimizer and other hyperparameters.
	# optimizer = optim.Adadelta(lstm.parameters(), lr=0.1)

	nEpochs = 10
	batchSize = 50

	#####################################################################################

	for epoch in range(nEpochs):
		totalLoss = 0.0

		(sentences_shuffled, tags_shuffled) = shuffle_sentences(sentences, tags)
		for batch in range(0, len(sentences), batchSize):
			#####################################################################################
			# TODO: Implement gradient update.

			loss = 0
			#####################################################################################

		print(f"loss on epoch {epoch} = {totalLoss}")
		lstm.write_predictions(sentences_dev, 'dev_pred')  # Performance on dev set
		print('conlleval:')
		print(subprocess.Popen('paste data/dev dev_pred | perl data/conlleval.pl -d "\t"', shell=True, stdout=subprocess.PIPE,
							   stderr=subprocess.STDOUT).communicate()[0].decode('UTF-8'))

		if epoch % 10 == 0:
			s = sample(range(len(sentences_dev)), 5)
			lstm.print_predictions([sentences_dev[i] for i in s], [tags_dev[i] for i in s])


if __name__ == "__main__":
	# Construct necessary dictionaries.
	GloVe = read_GloVe("data/glove.840B.300d.conll_filtered.txt")
	(sentences_dev, tags_dev)     = read_conll_format('data/dev')
	(sentences_train, tags_train) = read_conll_format('data/train')
	(sentences_test, tags_test)   = read_conll_format('data/test')
	sentencesChar = sentences2char(sentences_train)

	wordCounts = Counter([w for l in sentences_train for w in l])
	charCounts = Counter([c for l in sentences_train for w in l for c in w])
	singletons = set([w for (w,c) in wordCounts.items() if c == 1 and not w in GloVe.keys()])
	charSingletons = set([w for (w,c) in charCounts.items() if c == 1])

	word2i = {w:i+2 for i,w in enumerate(set([w for l in sentences_train for w in l] + list(GloVe.keys())))}
	char2i = {w:i+2 for i,w in enumerate(set([c for l in sentencesChar for w in l for c in w]))}
	i2word = {i:w for w,i in word2i.items()}
	i2char  = {i:w for w,i in char2i.items()}

	tag2i = {w:i for i,w in enumerate(set([t for l in tags_train for t in l]))}
	i2tag = {i:t for t,i in tag2i.items()}

	char_lstm = CharLSTMtagger(DIM_HID=500, DIM_EMB=300, tag2i=tag2i)
	train_char_lstm(sentences_train, tags_train, char_lstm)


