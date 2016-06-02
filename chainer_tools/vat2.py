# coding: utf-8

# In[1]:

import numpy as np
import chainer
import chainer.functions as F
from chainer import Variable, cuda

class InputGradientKeeper(chainer.Function):

    def __call__(self, inputs):
        self.init_gx(inputs)
        return super(InputGradientKeeper, self).__call__(inputs)

    def init_gx(self, inputs):
        xp = cuda.cupy.get_array_module(*inputs.data)
        self.gx = as_mat(xp.zeros_like(inputs.data))

    def forward(self, inputs):
        return inputs

    def backward(self, inputs, grad_outputs):
        self.gx.fill(0)
        self.gx += as_mat(grad_outputs[0])
        return grad_outputs

class CategoricalKLDivergence(chainer.Function):

    def __init__(self, unchain_py=True):
        self.unchain_py = unchain_py

    def forward(self, inputs):
        xp = cuda.cupy.get_array_module(*inputs[0])
        """
        return (1/N) * \sum_i^N \sum_j^L [py_ij * log(py_ij) - py_ij * log(py_tilde_ij)]
        """
        py,py_tilde = inputs
        kl = py * ( xp.log(py) - xp.log(py_tilde) )
        kl_sum = kl.sum(axis=1,keepdims=True)
        return kl_sum.mean(keepdims=True).reshape(()),


    def backward(self, inputs, grad_outputs):
        xp = cuda.cupy.get_array_module(*inputs[0])
        """
        (gradient w.r.t py) = log(py) + 1 - log(py_tilde)
        (gradient w.r.t py_tilde) = - py/py_tilde
        """
        py,py_tilde = inputs
        coeff = xp.asarray(grad_outputs[0]/py.shape[0],'float32')
        if(self.unchain_py):
            ret_py = None
        else:
            ret_py = coeff * ( xp.log(py) - xp.log(py_tilde) + 1.0)
        ret_py_tilde = -coeff * py/py_tilde
        return ret_py,ret_py_tilde



def categorical_kl_divergence(py, py_tilde, unchain_py=True):
    """Computes KL divergence between y and _y:KL[p(y|x)||p(_y|x)] (softmax activation only)

    Args:
        py (Variable): Variable holding a matrix whose (i, j)-th element
            indicates normalized probability of the class j at the i-th
            example.
        py_tilde (Variable): Variable holding a matrix whose (i, j)-th element
            indicates normalized probability of the class j at the i-th
            example (assumed to be probability y given "perturbed x").

    Returns:
        Variable: A variable holding a scalar array of the KL divergence loss.



    """
    return CategoricalKLDivergence(unchain_py)(py, py_tilde)



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
#        p, q = inputs
#        dq = -np.float32(1.0) * p / (np.float32(1e-8) + q) / np.float32(p.shape[0]) * grads[0]
#        return cuda.cupy.zeros_like(p), dq
        xp = cuda.cupy.get_array_module(*inputs[0])
        p,q = inputs
        dq = - p/q * xp.asarray(grads[0]/p.shape[0],'float32')

        return xp.zeros_like(p), dq

#def kl(p,q):
#    return KL_multinomial()(F.softmax(p),F.softmax(q))

def kl(p,q):
    #return KL_multinomial()(p,q)
    return categorical_kl_divergence(p,q)

def distance(y0, y1):
    return kl(F.softmax(y0), F.softmax(y1))

def as_mat(x):
    return x.reshape(x.shape[0], x.size // x.shape[0])
def normalize_axis1(x):
    xp = cuda.cupy.get_array_module(*x)
    abs_x = abs(x)
    x = x / (1e-6 + abs_x.max(axis=1,keepdims=True))
    x_norm_2 = x**2
    return x / xp.sqrt(1e-6 + x_norm_2.sum(axis=1,keepdims=True))
def perturbation_with_L2_norm_constraint(x,norm):
    return norm * normalize_axis1(x)
def perturbation_with_max_norm_constraint(x,norm):
    xp = cuda.cupy.get_array_module(*x)
    return norm * xp.sign(x)

def vat(forward, x, xi=1e-6, eps=1, Ip=1, device=0, test=False):
    x = cuda.to_gpu(x, device=device) 
    y = forward(Variable(x), test=test)
#    print x.sum(), y.data[0][0]

    #y.unchain_backward()
    xp = cuda.cupy.get_array_module(*x)
    d  = xp.random.normal(size=x.shape, dtype=np.float32)
    for ip in range(Ip):
        input_gradient_keeper = InputGradientKeeper()
        d       = normalize_axis1(as_mat(d)).reshape(x.shape)
        x_xi_d  = x + xi * Variable(d)
        x_xi_d_ = input_gradient_keeper(x_xi_d)
        y2      = forward(x_xi_d_, test=test)
        kl_loss = distance(y, y2)
        kl_loss.backward()
#        print x.sum(), x_xi_d_.data.sum()
#        print F.softmax(y).data[0][0],F.softmax(y2).data[0][0], kl_loss.data
        d = input_gradient_keeper.gx 
#        print d.sum()

#        quit()
    d     = normalize_axis1(as_mat(d)).reshape(x.shape)
    d_var = eps *Variable(d)
    y2    = forward(x +  d_var, test=test)

    return distance(y, y2)

