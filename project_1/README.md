# CS7650 - Project 1

This repository consists of starter code from the [Project 1 colab notebook](https://colab.research.google.com/drive/1QEPgC2FvXCvtZy2o4YBIkRPPLoG4gHJx). We setup the repo to only facilitate development outside the given colab notebook. Please note that you should submit the notebook and the output files to Gradescope and not this repository. The submission instructions are in the colab notebook.

In this assignment, you will implement the perceptron algorithm, and a simple, but competitive neural bag-of-words model, as described in [this](https://aclanthology.org/P15-1162.pdf) paper for text classification. You will train your models on a (provided) dataset of positive and negative movie reviews and report accuracy on a test set.


## Setting up the code

1. Please use python>=3.7.

2. Untar the data file.
  ```tar -xvzf aclImdb_small.tgz```
  
3. Download nltk punkt tokenizer. Run the following commands in the python interactive session.
```
import nltk
nltk.download('punkt')
```
## Running the code

You need to fill up the ```TODO``` sections in perceptron.py, nbow.py, utils.py, and cnn.py. Further instructions for each component are in the colab notebook.

To run perceptron code -

```python3 perceptron.py```

To run nbow/cnn code (without GPU) -

```python3 nbow.py / cnn.py```

To run nbow/cnn code (without GPU) -

```CUDA_VISIBLE_DEVICES=<GPU device id> python3 nbow.py / cnn.py```

