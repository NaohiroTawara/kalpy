
import numpy as np

import chainer
import chainer.functions as F
from chainer import FunctionSet, Variable, optimizers, cuda
from nn_parallel import NN_parallel

class KL_multinomial(chainer.function.Function):
    """ KL divergence between multinomial distributions """
    def __init__(self):
        pass
    def forward_gpu(self, inputs):
        p, q = inputs
        loss = cuda.cupy.ReductionKernel(
            'T p, T q',
            'T loss',
            'p*(log(p)-log(q))',
            'a + b',
            'loss = a',
            '0',
            'kl'
        )(p,q)
        return loss/np.float32(p.shape[0]),

    # backward only q-side
    def backward_gpu(self, inputs, grads):
        p, q = inputs
        dq = -np.float32(1.0) * p / (np.float32(1e-8) + q) / np.float32(p.shape[0]) * grads[0]
        return cuda.cupy.zeros_like(p), dq

def kl(p,q):
    return KL_multinomial()(F.softmax(p),F.softmax(q))

def kl(p,q):
    return KL_multinomial()(p,q)

def distance(y0, y1):
    return kl(F.softmax(y0), F.softmax(y1))
    
def vat(forward, distance, x, xi=10, eps=1.4, Ip=1, d_id=0):
    xp = cuda.cupy
    x = Variable(cuda.to_gpu(x, d_id))
    y = forward(x)
    y.unchain_backward()

    # calc adversarial direction
    d = xp.random.normal(size=x.data.shape, dtype=np.float32)
    d = d/xp.sqrt(xp.sum(d**2, axis=1)).reshape((x.data.shape[0],1))
    for ip in range(Ip):
        d_var = Variable(cuda.to_gpu(d.astype(np.float32), d_id))
        y2 = forward( x + xi * d_var)
        kl_loss = distance(y, y2)
        kl_loss.backward()
        d = d_var.grad
        d = d/xp.sqrt(xp.sum(d**2, axis=1)).reshape((x.data.shape[0],1))
    d_var = Variable(cuda.to_gpu(d.astype(np.float32), d_id))

    # calc regularization
    y2 = forward(x+eps*d_var)
    return distance(y, y2)


class NNetVAT(NN_parallel):
    def __init__(self, specs, njobs=-1):
        super(NNetVAT, self).__init__(specs, njobs)

    def loss_labeled(self, x, y, test=False, finetune=False):
        return self.loss_softmax(x,y,test,finetune)

    def loss_unlabeled(self, x):
        L = vat(self.forward_cupy, distance, x, d_id=self.device_id[0])
        return L

    def forward_cupy(self, x, test=False, finetune=False):
        return self.layers(x, test, finetune)   

    def loss_test(self, x, t):
        y = self.forward_cupy(x, test=True)
        L, acc = F.softmax_cross_entropy(y, t), F.accuracy(y, t)
        return L, acc


# Example of semi-supervised learning on MNIST
from load_mnist import *

if __name__ == "__main__":
    options_dict = {
        "layers_specs": [
            {"type": "full", "dimensions": (784, 1024), "activation": 'relu',
             'BN': (True),
             },
            {"type": "full", "dimensions": (1024, 1024), "activation": 'relu',
             'BN': (True),
             },
            {"type": "full", "dimensions": (1024, 10), "activation": 'linear',
             'BN': (True),
             },
            ],
        }
    model = NNetVAT(options_dict, njobs=1)
    model.set_optimizer(optimizers.Adam())
    model.initOptimizer()
    alpha_plan = [0.002] * 100
    for i in range(1,100):
        alpha_plan[i] = alpha_plan[i-1]*0.9
    batchsize_l = 100
    batchsize_ul = 250

    N_train_labeled = 100
    N_train_unlabeled = 60000
    N_test = 10000
    train_l, train_ul, test_set = \
        load_mnist(scale=1.0/128.0, shift=-1.0, N_train_labeled=N_train_labeled, \
                       N_train_unlabeled=N_train_unlabeled, N_test=N_test)
    for epoch in range(len(alpha_plan)):
#        print epoch

        sum_loss_l = 0
        sum_loss_ul = 0
        for it in range(60000/batchsize_ul):
            x,t = train_l.get(batchsize_l)
            loss_l = model.loss_labeled(x, t)
            
            model.zero_grads()
            loss_l.backward()
            model.update()

            x,_ = train_ul.get(batchsize_ul)
            loss_ul = model.loss_unlabeled(x)

            model.zero_grads()
            loss_ul.backward()
            model.update()

            sum_loss_l += loss_l.data
            sum_loss_ul += loss_ul.data

            loss_l.unchain_backward()
            loss_ul.unchain_backward()

        print "classification loss, vat loss: ", \
            sum_loss_l/(60000/batchsize_ul), sum_loss_ul/(60000/batchsize_ul)

#            o_enc.alpha = alpha_plan[epoch]

        x,t = test_set.get(10000, balance=False)
        L, acc = model.loss_test(Variable(x, volatile=True), Variable(t, volatile=True))
        x,t = train_l.get(100)
        L_, acc_ = model.loss_test(Variable(x), Variable(t))
        L_.unchain_backward()
        acc_.unchain_backward()
        print "test error, test acc, train error, train acc: ", L.data, acc.data, L_.data, acc_.data
        sys.stdout.flush()

