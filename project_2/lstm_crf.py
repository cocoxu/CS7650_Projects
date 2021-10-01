import torch.nn.functional as F

#####################################################################################
# TODO: Add imports if needed.
import os
import random
import subprocess
import torch.nn as nn
from random import sample

from preprocess import *
from read_data import *
from basic_lstm import *
from char_lstm import *

#####################################################################################

class LSTM_CRFtagger(CharLSTMtagger):
	def __init__(self, DIM_EMB=10, DIM_CHAR_EMB=30, DIM_HID=10, tag2i=None):

		N_TAGS = max(tag2i.values()) + 1
		super(LSTM_CRFtagger, self).__init__(DIM_EMB=DIM_EMB, DIM_HID=DIM_HID, DIM_CHAR_EMB=DIM_CHAR_EMB, tag2i=tag2i)
		#####################################################################################
		# TODO: Initialize parameters.

		self.transitionWeights = nn.Parameter(torch.zeros((N_TAGS, N_TAGS), requires_grad=True))
		nn.init.normal_(self.transitionWeights)

		#####################################################################################

	def gold_score(self, lstm_scores, Y):
		#####################################################################################
		# TODO: compute score of gold sequence Y (unnormalized conditional log-probability)

		return 0

		#####################################################################################


	# Forward algorithm for a single sentence
	# Efficiency will eventually be important here.  We recommend you start by
	# training on a single batch and make sure your code can memorize the
	# training data.  Then you can go back and re-write the inner loop using
	# tensor operations to speed things up.
	def forward_algorithm(self, lstm_scores, sLen):
		#####################################################################################
		# TODO: implement forward algorithm.
		return 0

		#####################################################################################

	def conditional_log_likelihood(self, sentences, tags, train=True):
		#####################################################################################
		# TODO: compute conditional log likelihood of Y (use forward_algorithm and gold_score)
		return 0

		#####################################################################################

	def viterbi(self, lstm_scores, sLen):
		#####################################################################################
		# TODO: Implement Viterbi algorithm, soring backpointers to recover the argmax sequence.  Returns the argmax sequence in addition to its unnormalized conditional log-likelihood.
		return (torch.as_tensor([random.randint(0, lstm_scores.shape[1] - 1) for x in range(sLen)]), 0)

		#####################################################################################

	# Computes Viterbi sequences on a batch of data.
	def viterbi_batch(self, sentences):
		viterbiSeqs = []
		(X, X_mask, X_char) = self.sentences2input_tensors(sentences)
		lstm_scores = self.forward(X, X_char)
		for s in range(len(sentences)):
			(viterbiSeq, ll) = self.viterbi(lstm_scores[s], len(sentences[s]))
			viterbiSeqs.append(viterbiSeq)
		return viterbiSeqs

	def forward(self, X, X_char, train=False):
		#####################################################################################
		# TODO: Implement the forward computation.

		return torch.randn((X.shape[0], X.shape[1], self.NUM_TAGS))  # Random baseline.

		#####################################################################################

	def sentences2input_tensors(self, sentences):
		(X, X_mask)   = prepare_input(sentences2indices(sentences, word2i))
		X_char        = prepare_input_char(sentences2indicesChar(sentences, char2i))
		return (X, X_mask, X_char)

	def print_predictions(self, words, tags):
		Y_pred = self.inference(words)
		for i in range(len(words)):
			print("----------------------------")
			print(" ".join([f"{words[i][j]}/{Y_pred[i][j]}/{tags[i][j]}" for j in range(len(words[i]))]))
			print("Predicted:\t", [Y_pred[i][j] for j in range(len(words[i]))])
			print("Gold:\t\t", tags[i])

	# Need to use Viterbi this time.
	def inference(self, sentences, viterbi=True):
		pred = self.viterbi_batch(sentences)
		return [[i2tag[pred[i][j].item()] for j in range(len(sentences[i]))] for i in range(len(sentences))]


# CharLSTM-CRF Training

def train_crf_lstm(sentences, tags, lstm):
	#####################################################################################
	# TODO: initialize optimizer and hyperparameters.
	# optimizer = optim.Adadelta(lstm.parameters(), lr=1.0)

	nEpochs = 10
	batchSize = 50
	#####################################################################################

	for epoch in range(nEpochs):
		totalLoss = 0.0
		lstm.train()

		# Shuffle the sentences
		(sentences_shuffled, tags_shuffled) = shuffle_sentences(sentences, tags)
		for batch in range(0, len(sentences), batchSize):
			#####################################################################################
			# TODO: Implement gradient update on a batch of data.

			pass
			#####################################################################################

		print(f"loss on epoch {epoch} = {totalLoss}")
		lstm.write_predictions(sentences_dev, 'dev_pred')  # Performance on dev set
		print('conlleval:')
		print(subprocess.Popen('paste data/dev dev_pred | perl data/conlleval.pl -d "\t"', shell=True, stdout=subprocess.PIPE,
							   stderr=subprocess.STDOUT).communicate()[0].decode('UTF-8'))

		if epoch % 10 == 0:
			lstm.eval()
			s = random.sample(range(50), 5)
			lstm.print_predictions([sentences_train[i] for i in s],
								   [tags_train[i] for i in s])  # Print predictions on train data (useful for debugging)


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

	global word2i
	word2i = {w:i+2 for i,w in enumerate(set([w for l in sentences_train for w in l] + list(GloVe.keys())))}
	char2i = {w:i+2 for i,w in enumerate(set([c for l in sentencesChar for w in l for c in w]))}
	i2word = {i:w for w,i in word2i.items()}
	i2char  = {i:w for w,i in char2i.items()}

	tag2i = {w:i for i,w in enumerate(set([t for l in tags_train for t in l]))}
	i2tag = {i:t for t,i in tag2i.items()}

	crf_lstm = LSTM_CRFtagger(DIM_HID=500, DIM_EMB=300, DIM_CHAR_EMB=30, tag2i=tag2i)
	train_crf_lstm(sentences_train, tags_train, crf_lstm)  # Train on the full dataset
	# train_crf_lstm(sentences_train[0:50], tags_train[0:50])          #Train only the first batch (use this during development/debugging)
