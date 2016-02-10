# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 18:57:42 2015

@author: tawara
"""
from chainer import FunctionSet, Variable, cuda, optimizers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

import cupy
import copy
import numpy as np

import my_batch_normalization
reload(my_batch_normalization)
from my_batch_normalization import BatchNormalization
from max_unpooling_2d import max_unpooling_2d

# # Define NN
class NN_parallel():
    def __init__(self, specs, njobs):
        self.specs = specs
        assert specs.has_key("layers_specs"), "Please specify specs of each layer"
        layers_specs = self.specs["layers_specs"]
        self.layers = CNN(layers_specs)
        #rev_layers_specs = copy.deepcopy([spec for spec in reversed(specs["layers_specs"])])
        #for spec in rev_layers_specs:
        #    if spec["type"] == "full":
        #        spec["dimensions"] = (spec["dimensions"][1], spec["dimensions"][0])
        #    elif spec["type"] == "conv":
        #        spec["type"] = "deconv"
        #        spec["filter_shape"] = (spec["filter_shape"][1], spec["filter_shape"][0], spec["filter_shape"][2], spec["filter_shape"][3])
        #self.dec_layers= CNN(rev_layers_specs)        
        if njobs==-1:
            self.device_id = [-1]
        else:
            self.device_id = [4]
            self.layers.to_gpu(self.device_id[0])

        if self.specs["learning_rule"]["type"] == "adam":
            self.optimizer = optimizers.Adam()
        elif self.specs["learning_rule"]["type"] == "adadelta":
            self.optimizer = optimizers.AdaDelta()
        elif self.specs["learning_rule"]["type"] == "momentum":
            self.optimizer = optimizers.MomentumSGD()
        else:
            raise ValueError("Unsupported rule" + str(self.specs["learning_rule"]["type"]))
        self.optimizer.setup(self.layers)

    def update(self):
        self.optimizer.update()

    def zero_grads(self):
        self.layers.zerograds()

    def forward(self, x, test=False, finetune=False):
        if self.device_id[0] >=0:
            x = cuda.to_gpu(x, device=self.device_id[0])
        return self.layers(Variable(x), test, finetune)

    def loss_softmax(self, x, y, test=False, finetune=False):
        if self.device_id[0] >=0:
            y = cuda.to_gpu(y, device=self.device_id[0])
        return F.softmax_cross_entropy(self.forward(x, test,finetune), Variable(y))

#-------------------------------------------------------------------
#-------------------------------------------------------------------
#-------------------------------------------------------------------

class CNN(ChainList):
    def __init__(self, specs):
        assert type(specs)==list, \
            "Specs must be a list (Not " + str(type(specs)) + ")\n" +  str(specs)
        self.specs = specs
        self.n_layers = len(specs)
        layers=[]
        for spec in specs:
            if spec["type"] == "conv":
                layers.append(ConvLayer(spec))
            elif spec["type"] == "full":
                layers.append(HiddenLayer(spec))
            elif spec["type"] == "BN":
                assert spec.has_key("dimensions"), "Size of channel dimensions is required"
                layers.append(BatchNormalization(spec["dimensions"]))
            elif spec["type"] == "deconv":
                layers.append(ConvLayer(spec, isDecoding=True))
            else:
                raise ValueError("Unsupported layer" + spec["type"])
        super(CNN, self).__init__(*layers)

    def __call__(self, x, test, upd_batch_est):
        h = self[0](x, test, upd_batch_est)
        for i in xrange(1, self.n_layers):
            h = self[i](h, test, upd_batch_est)
        return h

    def deep_copyparams(self, cnn):
        for i in xrange(0, self.n_layers):
            self[i].copyparams(cnn[i])

class Layer():
    def __apply_activation__(self, h):
        if self.specs.has_key("activation"):
            if self.specs["activation"] == "relu":
                h = F.relu(h)
            elif self.specs["activation"] == "linear":
                h = h
            elif self.specs["activation"] == "sigmoid":
                h = F.sigmoid(h)
            else:
                raise ValueError("Unsupported activation function: " + self.specs["activation"])
        return h


class ConvLayer(Chain, Layer):
    def __init__(self, specs, isDecoding=False):
        self.specs = specs
        self.is_decoding = isDecoding
        assert self.specs.has_key("filter_shape"), \
            "Please specify filter shape: (out_channels, in_channels, ksize_x, ksize_y)\n" + \
            "Current specs is "+ str(specs)
        out_channels =  self.specs["filter_shape"][0]
        in_channels  =  self.specs["filter_shape"][1]
        ksize        = (self.specs["filter_shape"][2], self.specs["filter_shape"][3])
        if self.is_decoding:
            conv = L.Deconvolution2D(in_channels, out_channels, ksize)
        else:
            conv = L.Convolution2D(in_channels, out_channels, ksize)
        if self.specs.has_key("BN"):
            assert len(self.specs["BN"]) == 2, \
                "argument of BN must be ( {True or False}, {True or False} ). " + \
                "The first element corresponds to the output of activation function " + \
                "and the second element corresponds to the output of pooling function (if it exists)."
            if sum(self.specs["BN"]) == 1:
                super(ConvLayer, self).__init__(
                    conv = conv,
                    bn1  = BatchNormalization(out_channels)
                )
            elif sum(self.specs["BN"]) == 2:
                assert self.specs.has_key("pool_shape"), \
                    "Pooling layer is required if batch normalization is used after pooling function"
                super(ConvLayer, self).__init__(
                    conv = conv,
                    bn1  = BatchNormalization(out_channels),
                    bn2  = BatchNormalization(out_channels)
                )
        else:
            super(ConvLayer, self).__init__(
                conv = conv
            )

    def __call__(self, x, test, finetune):
        if self.is_decoding:
            if self.specs.has_key("pool_shape"):
                # Reshape concatinated vector from the previous hidden layer into image
                if len(x.data.shape) == 2:
                    assert self.specs.has_key("d_out"), \
                        "Please specify output dimension (This will be automaticaly estimated in the future version)"
                    x = F.reshape(x,(x.data.shape[0], self.specs["filter_shape"][1], 
                                     self.specs["d_out"][0], self.specs["d_out"][1]))
                d_out = (x.data.shape[2] * self.specs["pool_shape"][0],
                         x.data.shape[3] * self.specs["pool_shape"][1])
                h = max_unpooling_2d(x, d_out, self.specs["pool_shape"])
            h = self.conv(h)
            h = self.__apply_activation__(h)
        else:
            # So far, batch normalization is only implemented for encoding network
            h = self.conv(x)
            if self.specs.has_key("BN") and self.specs["BN"][0]:
                h = self.bn1(h, test, finetune)
            h = self.__apply_activation__(h)
            if self.specs.has_key("pool_shape"):
                h = F.max_pooling_2d(h, self.specs["pool_shape"])
            if self.specs.has_key("BN") and self.specs["BN"][1]:
                if self.specs["BN"][0]:
                    h = self.bn2(h, test, finetune)
                else:
                    h = self.bn1(h, test, finetune)
        return h
#        return F.max_pooling_2d(F.relu(self.conv(x)), self.specs["pool_shape"])


class HiddenLayer(Chain, Layer):
    def __init__(self, specs):
        self.specs = specs
        assert self.specs.has_key("dimensions"), "Please specify dimensions (input, output)"
        in_dim, out_dim = self.specs["dimensions"]
        if self.specs.has_key("BN") and self.specs["BN"]:
            super(HiddenLayer, self).__init__(
                ln = L.Linear(in_dim, out_dim),
                bn = BatchNormalization(out_dim)
            )
        else:
            super(HiddenLayer, self).__init__(
                ln = L.Linear(in_dim, out_dim)
            )
    def __call__(self,x, test, finetune):
        h = self.ln(x)
        if self.specs.has_key("BN") and self.specs["BN"]:
            h = self.bn(h, test, finetune)
        h = self.__apply_activation__(h)
        if self.specs.has_key("dropout") :
            assert self.specs["dropout"]>0 and self.specs["dropout"] <=1, \
                "Dropout ratio must be within (0,1]"
            h = F.dropout(h, self.specs["dropout"], train = not test)
        return h



def main():
    npz = np.load("/home-nfs/tawara/work/ttic/data/icassp15.0/swbd.test.npz")
    utt_ids = sorted(npz.keys())
    mats = np.array([npz[i] for i in utt_ids])
    mats = np.array([[t] for t in mats]).astype(np.float32)
