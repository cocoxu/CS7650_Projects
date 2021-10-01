# Read in the training data.
def read_conll_format(filename):
	(words, tags, currentSent, currentTags) = ([],[],['-START-'],['START'])
	for line in open(filename).readlines():
		line = line.strip()
		if line == "":
			currentSent.append('-END-')
			currentTags.append('END')
			words.append(currentSent)
			tags.append(currentTags)
			(currentSent, currentTags) = (['-START-'], ['START'])
		else:
			(word, tag) = line.split()
			currentSent.append(word)
			currentTags.append(tag)
	return (words, tags)


# Read GloVe embeddings.
def read_GloVe(filename):
	embeddings = {}
	for line in open(filename).readlines():
		fields = line.strip().split(" ")
		word = fields[0]
		embeddings[word] = [float(x) for x in fields[1:]]
	return embeddings