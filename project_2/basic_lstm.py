#####################################################################################
#TODO: Add imports if needed:

#####################################################################################

import subprocess
import torch.nn as nn
from preprocess import *
from read_data import *
from random import sample


class BasicLSTMtagger(nn.Module):
	def __init__(self, DIM_EMB=10, DIM_HID=10, tag2i=None):
		super(BasicLSTMtagger, self).__init__()
		NUM_TAGS = max(tag2i.values()) + 1


		(self.DIM_EMB, self.NUM_TAGS) = (DIM_EMB, NUM_TAGS)
	#####################################################################################
	#TODO: initialize parameters - embedding layer, nn.LSTM, nn.Linear and nn.LogSoftmax

	#####################################################################################


	def forward(self, X, X_mask, train=False):
		#####################################################################################
		#TODO: Implement the forward computation.
		#X is padded. The predictions corresponding to the padded positions need to removed to calculate loss correctly.

		return torch.randn((X.shape[0], X.shape[1], self.NUM_TAGS))  #Random baseline.
	#####################################################################################

	def init_glove(self, GloVe):
		#####################################################################################
		#TODO: initialize word embeddings using GloVe (you can skip this part in your first version, if you want, see instructions below).
		pass

	#####################################################################################

	def inference(self, sentences):
		X, X_mask = prepare_input(sentences2indices(sentences, word2i))
		pred = self.forward(X, X_mask).argmax(dim=2)
		return [[i2tag[pred[i,j].item()] for j in range(len(sentences[i]))] for i in range(len(sentences))]

	def print_predictions(self, words, tags):
		Y_pred = self.inference(words)
		for i in range(len(words)):
			print("----------------------------")
			print(" ".join([f"{words[i][j]}/{Y_pred[i][j]}/{tags[i][j]}" for j in range(len(words[i]))]))
			print("Predicted:\t", Y_pred[i])
			print("Gold:\t\t", tags[i])

	def write_predictions(self, sentences, outFile):
		fOut = open(outFile, 'w')
		for s in sentences:
			y = self.inference([s])[0]
			#print("\n".join(y[1:len(y)-1]))
			fOut.write("\n".join(y[1:len(y)-1]))  #Skip start and end tokens
			fOut.write("\n\n")

# Training

def shuffle_sentences(sentences, tags):
	shuffled_sentences = []
	shuffled_tags = []
	indices = list(range(len(sentences)))
	random.shuffle(indices)
	for i in indices:
		shuffled_sentences.append(sentences[i])
		shuffled_tags.append(tags[i])
	return (shuffled_sentences, shuffled_tags)


def train_basic_lstm(sentences, tags, lstm):
	#####################################################################################
	# TODO: initialize optimizer and other hyperparameters.
	# optimizer = optim.Adadelta(lstm.parameters(), lr=0.1)

	batchSize = 50
	nEpochs = 10
	#####################################################################################

	for epoch in range(nEpochs):
		totalLoss = 0.0

		(sentences_shuffled, tags_shuffled) = shuffle_sentences(sentences, tags)
		for batch in range(0, len(sentences), batchSize):
			#####################################################################################
			# TODO: Implement gradient update.


			loss = 0
			totalLoss += loss
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

	lstm = BasicLSTMtagger(DIM_HID=500, DIM_EMB=300, tag2i=tag2i)
	train_basic_lstm(sentences_train, tags_train, lstm)
