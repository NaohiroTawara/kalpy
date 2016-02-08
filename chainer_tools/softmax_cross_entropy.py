import numpy
import six

from chainer import cuda
from chainer import function
from chainer.utils import type_check


def logsumexp(x):
    xp = cuda.get_array_module(x)
    m = x.max(axis=1, keepdims=True)
    y = x - m
    xp.exp(y, out=y)
    return xp.log(y.sum(axis=1, keepdims=True)) + m


def softmax_log(x):
    # TODO(unno): Use cudnn (cudnn v2 doesn't support CUDNN_SOFTMAX_LOG)
    log_z = logsumexp(x)
    return x - log_z


class SoftmaxCrossEntropy(function.Function):

    """Softmax activation followed by a cross entropy loss."""

    ignore_label = -1

    def __init__(self, use_cudnn=True, normalize=True):
        self.use_cudnn = use_cudnn
        self.normalize = normalize

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        x_type, t_type = in_types

        type_check.expect(
            x_type.dtype == numpy.float32,
            t_type.dtype == numpy.int32,
            t_type.ndim == x_type.ndim - 1,

            x_type.shape[0] == t_type.shape[0],
            x_type.shape[2:] == t_type.shape[1:],
        )

    def forward_cpu(self, inputs):
        x, t = inputs
        log_y = softmax_log(x)
        self.y = numpy.exp(log_y)
        log_yd = numpy.rollaxis(log_y, 1)
        log_yd = log_yd.reshape(len(log_yd), -1)

        log_p = log_yd[numpy.maximum(t.flat, 0), six.moves.range(t.size)]
        # deal with the case where the SoftmaxCrossEntropy is
        # unpickled from the old version
        if getattr(self, 'normalize', True):
            count = (t != self.ignore_label).sum()
        else:
            count = x.shape[0]
        self.count = count

        if count == 0:
            return numpy.zeros((), dtype=x.dtype),

        y = (log_p * (t.flat != self.ignore_label)).sum(keepdims=True) \
            * (-1.0 / count)
        return y.reshape(()),

    def forward_gpu(self, inputs):
        cupy = cuda.cupy
        x, t = inputs
        log_y = softmax_log(x)
        self.y = cupy.exp(log_y)
        if getattr(self, 'normalize', True):
            count = float((t != self.ignore_label).sum())
        else:
            count = t.shape[0]
        self.count = count

        if count == 0:
            return cupy.zeros((), dtype=x.dtype),

        log_y = cupy.rollaxis(log_y, 1, log_y.ndim)
        ret = cuda.reduce(
            'S t, raw T log_y, int32 n_channel, T inv_count', 'T out',
            't == -1 ? 0 : log_y[_j * n_channel + t]',
            'a + b', 'out = a * inv_count', '0', 'crossent_fwd'
        )(t, log_y.reduced_view(), log_y.shape[-1], -1.0 / count)
        return ret,

    def backward_cpu(self, inputs, grad_outputs):
        x, t = inputs
        if self.count == 0:
            return numpy.zeros_like(x), None

        gloss = grad_outputs[0]
        n_unit = t.size // t.shape[0]
        if self.y.ndim == 2:
            gx = self.y.copy()
            gx[six.moves.xrange(len(t)), numpy.maximum(t, 0)] -= 1
            gx *= (t != self.ignore_label).reshape((len(t), 1))
        else:
            # in the case where y.ndim is higher than 2,
            # we think that a current implementation is inefficient
            # because it yields two provisional arrays for indexing.
            gx = self.y.copy().reshape(self.y.shape[0], self.y.shape[1], -1)
            fst_index = numpy.arange(t.size) // n_unit
            trd_index = numpy.arange(t.size) % n_unit
            gx[fst_index, numpy.maximum(t.flat, 0), trd_index] -= 1
            gx *= (t != self.ignore_label).reshape((len(t), 1, -1))
            gx = gx.reshape(self.y.shape)

        gx *= gloss / self.count
        return gx, None

    def backward_gpu(self, inputs, grad_outputs):
        cupy = cuda.cupy
        x, t = inputs
        if self.count == 0:
            return cupy.zeros_like(x), None

        gloss = grad_outputs[0]
        n_unit = t.size // t.shape[0]
        coeff = cuda.cupy.divide(gloss, self.count, dtype=gloss.dtype)
        gx = cuda.elementwise(
            'T y, S t, raw T coeff, S n_channel, S n_unit',
            'T gx',
            '''
               const int c = (i / n_unit % n_channel);
               if (t == -1) {
                 gx = 0;
               } else {
                 gx = coeff[0] * (y - (c == t));
               }
            ''',
            'softmax_crossent_bwd')(
                self.y, cupy.expand_dims(t, 1), coeff, x.shape[1], n_unit)
        return gx, None


def softmax_cross_entropy(x, t, use_cudnn=True, normalize=True):
    """Computes cross entropy loss for pre-softmax activations.

    Args:
        x (Variable): Variable holding a multidimensional array whose element
            indicates unnormalized log probability: the first axis of the
            variable represents the number of samples, and the second axis
            represents the number of classes. While this function computes
            a usual softmax cross entropy if the number of dimensions is equal
            to 2, it computes a cross entropy of the replicated softmax if the
            number of dimensions is greater than 2.
        t (Variable): Variable holding an int32 vector of groundtruth labels.
            If ``t[i] == -1``, correspondig ``x[i]`` is ignored.
        normalize (Variable): Variable holding a boolean value which
            determines the normalization constant. If true, this function
            normalizes the cross entropy loss across all instances. If else,
            it only normalizes along a batch size.

    Returns:
        Variable: A variable holding a scalar array of the cross entropy loss.

    .. note::

       This function is differentiable only by ``x``.

    """
    return SoftmaxCrossEntropy(use_cudnn, normalize)(x, t)
