# kalpy

Python scripts for Chainer(<http://chainer.org/>) and Kaldi(<http://kaldi.sourceforge.net>).

## Preparation

### Install nvidia-ml-py
```
wget  --no-check-certificate https://pypi.python.org/packages/source/n/nvidia-ml-py/nvidia-ml-py-7.352.0.tar.gz ./
tar xzvf nvidia-ml-py-7.352.0.tar.gz
cd nvidia-ml-py-7.352.0
python setup.py
```

### Install kaldi and chainer
Follow the instructions of the each manual.

## Description
Kalpy is a set of python scripts for manupulating Chainer(<http://chainer.org/>) and Kaldi(<http://kaldi.sourceforge.net>).

- chainer_tools
-- nn_parallel.py implements a chainer-based NeuralNetwork that supports following structure
   --- Model-based parallelization
   --- Convolutional-net, Batch

-cuda_tools
--  cuda_utils.py contains utility functions for nvidia-ml-py.

-kaldi
Scripts in this directly implements data stream functions that provides KALDI-format (e.x. ark, scp, nnet, ali, etc.) in python-friendly format (e.x. npz, HDF5).
-- commands.py implements data stream functions for the Kaldi-format that directly utializes KALDI commands.
-- io.py implements data stream functions for the Kaldi-format that does NOT depend on KALDI commands.
-- data.py implements wrapper functions to read/write use Kaldi-format.

-timit
These files is a simple example of kaldi-based phonerecognition for TIMIT dataset.
They are basically based on egs/timit/s5 recipe.
