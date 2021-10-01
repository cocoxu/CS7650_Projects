# CS7650 - Project 2

This repository consists of starter code from the [Project 2 colab notebook](https://colab.research.google.com/drive/195wqm4BSTagmBmckfNVMv17n9e_qMCn_). 
We setup the repo to only facilitate development outside the given colab notebook. 
Please note that you should submit the notebook and the output files to Gradescope and not this repository. 
The submission instructions are in the colab notebook.

## Setting up the code

1. Please use python>=3.7.

2. Extract the data file.
  ```tar -xvzf data.tgz```
  
## Running the code

You need to fill up the ```TODO``` sections in basic_lstm.py, char_lstm.py, and lstm_crf.py.
Further instructions for each component are in the colab notebook.

To run basic lstm tagger -

```python3 basic_lstm.py```

To run char lstm tagger -

```python3 char_lstm.py```

To run crf tagger -

```python3 lstm_crf.py```

If you want to run with GPU, please use ```CUDA_VISIBLE_DEVICES=<GPU device id>''' before the command.


