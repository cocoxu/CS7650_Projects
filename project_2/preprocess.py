# Input and sentence preprocessing functions.

import torch
import random
from collections import Counter
from read_data import read_GloVe, read_conll_format


# When training, randomly replace singletons with UNK tokens sometimes to simulate situation at test time.
def getDictionaryRandomUnk(w, dictionary, train=False):
	if train and (w in singletons and random.random() > 0.5):
		return 1
	else:
		return dictionary.get(w, 1)


# Converts sentences to character sequences of words.
def sentences2char(sentences):
	return [[['start'] + [c for c in w] + ['end'] for w in l] for l in sentences]


# Map a list of sentences from words to indices.
def sentences2indices(words, dictionary, train=False):
	# 1.0 => UNK
	return [[getDictionaryRandomUnk(w, dictionary, train=train) for w in l] for l in words]


# Map a list of sentences containing to indices (character indices)
def sentences2indicesChar(chars, dictionary):
	# 1.0 => UNK
	return [[[dictionary.get(c, 1) for c in w] for w in l] for l in chars]


# Pad inputs to max sequence length (for batching)
def prepare_input(X_list):
	X_padded = torch.nn.utils.rnn.pad_sequence([torch.as_tensor(l) for l in X_list], batch_first=True).type(torch.LongTensor)
	X_mask = torch.nn.utils.rnn.pad_sequence([torch.as_tensor([1.0] * len(l)) for l in X_list], batch_first=True).type(torch.FloatTensor)
	return X_padded, X_mask


# Maximum word length (for character representations)
MAX_CLEN = 32


def prepare_input_char(X_list):
	MAX_SLEN = max([len(l) for l in X_list])
	X_padded = [l + [[]]*(MAX_SLEN-len(l))  for l in X_list]
	X_padded = [[w[0:MAX_CLEN] for w in l] for l in X_padded]
	X_padded = [[w + [1]*(MAX_CLEN-len(w)) for w in l] for l in X_padded]
	return torch.as_tensor(X_padded).type(torch.LongTensor)


# Pad outputs using one-hot encoding
def prepare_output_onehot(Y_list, NUM_TAGS):
	Y_onehot = [torch.zeros(len(l), NUM_TAGS) for l in Y_list]
	for i in range(len(Y_list)):
		for j in range(len(Y_list[i])):
			Y_onehot[i][j,Y_list[i][j]] = 1.0
	Y_padded = torch.nn.utils.rnn.pad_sequence(Y_onehot, batch_first=True).type(torch.FloatTensor)
	return Y_padded


if __name__ == "__main__":
	GloVe = read_GloVe("data/glove.840B.300d.conll_filtered.txt")
	(sentences_train, tags_train) = read_conll_format("data/train")
	(sentences_dev, tags_dev) = read_conll_format("data/dev")
	sentencesChar = sentences2char(sentences_train)

	print("\n")
	print("Training data example:")
	print(sentences_train[2])
	print(tags_train[2])

	print("\n")
	print("Glove vector example:")
	print(GloVe['the'])

	# Will need this later to remove 50% of words that only appear once in the training data from the vocabulary
	# (and don't have GloVe embeddings).
	wordCounts = Counter([w for l in sentences_train for w in l])
	charCounts = Counter([c for l in sentences_train for w in l for c in w])
	singletons = set([w for (w,c) in wordCounts.items() if c == 1 and not w in GloVe.keys()])
	charSingletons = set([w for (w,c) in charCounts.items() if c == 1])

	# Build dictionaries to map from words, characters to indices and vice versa.
	# Save first two words in the vocabulary for padding and "UNK" token.
	word2i = {w:i+2 for i,w in enumerate(set([w for l in sentences_train for w in l] + list(GloVe.keys())))}
	char2i = {w:i+2 for i,w in enumerate(set([c for l in sentencesChar for w in l for c in w]))}
	i2word = {i:w for w,i in word2i.items()}
	i2char  = {i:w for w,i in char2i.items()}

	# Tag dictionaries.
	tag2i = {w:i for i,w in enumerate(set([t for l in tags_train for t in l]))}
	i2tag = {i:t for t,i in tag2i.items()}

	# Indices
	X = sentences2indices(sentences_train, word2i, train=True)
	X_char = sentences2indicesChar(sentencesChar, char2i)
	Y = sentences2indices(tags_train, tag2i)

	print("\n")
	print("Index to vocab word example:")
	print(253, i2word[253])

	print("\n")
	print("Few Dev examples:")
	# Print out some examples of what the dev inputs will look like
	for i in range(10):
		print(" ".join([i2word.get(w, 'UNK') for w in X[i]]))

	print("\n")
	print("max slen:", max([len(x) for x in X_char]))  #Max sequence length in the training data is 39.

	(X_padded, X_mask) = prepare_input(X)
	X_padded_char = prepare_input_char(X_char)
	Y_onehot = prepare_output_onehot(Y, max(tag2i.values())+1)

	print("\n")
	print("X_padded:", X_padded.shape)
	print("X_padded_char:", X_padded_char.shape)
	print("Y_onehot:", Y_onehot.shape)

