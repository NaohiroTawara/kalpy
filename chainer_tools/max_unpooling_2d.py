import numpy

from chainer import cuda
from chainer.functions.pooling import pooling_2d
from chainer.utils import conv

if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    libcudnn = cudnn.cudnn


class MaxUnPooling2D(pooling_2d.Pooling2D):
    def __init__(self, d_out, ksize, stride=None, pad=0, cover_all=True,
                 use_cudnn=True):
        self.h, self.w = d_out
        super(MaxUnPooling2D, self).__init__(ksize, stride, pad, cover_all,use_cudnn)

    """Max un-pooling over a set of 2d planes."""

    def backward_cpu(self, x, gy):
        # x is a dummy variable, which is required only for compatibility with pooling_2d.Pooling2D
        col = conv.im2col_cpu(
            gy[0], self.kh, self.kw, self.sy, self.sx, self.ph, self.pw,
            pval=-float('inf'), cover_all=self.cover_all)
        n, c, kh, kw, out_h, out_w = col.shape
        col = col.reshape(n, c, kh * kw, out_h, out_w)

        # We select maximum twice, since the implementation using numpy.choose
        # hits its bug when kh * kw >= 32.
        gx = col.max(axis=2)
        return gx,


    def backward_gpu(self, x, gy):
        # x is a dummy variable, which is required only for compatibility with pooling_2d.Pooling2D
        n, c, h, w = gy[0].shape
        y_h = conv.get_conv_outsize(
            h, self.kh, self.sy, self.ph, self.cover_all)
        y_w = conv.get_conv_outsize(
            w, self.kw, self.sx, self.pw, self.cover_all)
        y = cuda.empty((n, c, y_h, y_w), dtype=gy[0].dtype)
        gx = cuda.empty((n, c, y_h, y_w), dtype=x[0].dtype)
        cuda.elementwise(
            'raw T in, int32 h, int32 w, int32 out_h, int32 out_w,'
            'int32 kh, int32 kw, int32 sy, int32 sx, int32 ph, int32 pw',
            'T out',
            '''
               int c0    = i / (out_h * out_w);
               int out_y = i / out_w % out_h;
               int out_x = i % out_w;
               int in_y_0 = max(0, out_y * sy - ph);
               int in_y_1 = min(h, out_y * sy + kh - ph);
               int in_x_0 = max(0, out_x * sx - pw);
               int in_x_1 = min(w, out_x * sx + kw - pw);

               float maxval = in[in_x_0 + w * (in_y_0 + h * c0)];
               int argmax_y = in_y_0;
               int argmax_x = in_x_0;
               for (int y = in_y_0; y < in_y_1; ++y) {
                 int offset_y = w * (y + h * c0);
                 for (int x = in_x_0; x < in_x_1; ++x) {
                   float v = in[x + offset_y];
                   if (maxval < v) {
                     maxval   = v;
                     argmax_y = y;
                     argmax_x = x;
                   }
                 }
               }
               out = maxval;

               int argmax_ky = argmax_y + ph - out_y * sy;
               int argmax_kx = argmax_x + pw - out_x * sx;
            ''', 'max_pool_fwd')(gy[0].reduced_view(),
                                 h, w, y_h, y_w, self.kh, self.kw,
                                 self.sy, self.sx, self.ph, self.pw,
                                 gx)
        return gx,

    def forward_cpu(self, x):
        n, c, out_h, out_w = x[0].shape
        gcol = numpy.zeros(
            (n, c, self.kh, self.kw, out_h, out_w), dtype=numpy.float32)

        # TODO(beam2d): Make it fast
        gcol_r = numpy.rollaxis(gcol.reshape(n, c, -1, out_h, out_w), 2)
        for i in numpy.ndindex(n, c, out_h, out_w):
#            gcol_r[self.indexes[i]][i] = x[0][i]
            for j in xrange(gcol_r.shape[0]):
                gcol_r[j][i] = x[0][i]

        y = conv.col2im_cpu(gcol, self.sy, self.sx, self.ph, self.pw, self.h, self.w)
        return y,

    def forward_gpu(self, x):
        n, c, y_h, y_w = x[0].shape
        gx = cuda.empty_like(numpy.ones((n, c, self.h,self.w)).astype(numpy.float32))
        cuda.elementwise(
            'raw T gy, int32 h, int32 w,'
            'int32 out_h, int32 out_w, int32 kh, int32 kw,'
            'int32 sy, int32 sx, int32 ph, int32 pw',
            'T gx',
            '''
               int c0 = i / (h * w);
               int y  = i / w % h + ph;
               int x  = i % w + pw;
               int out_y_0 = max(0,     (y - kh + sy) / sy);
               int out_y_1 = min(out_h, (y      + sy) / sy);
               int out_x_0 = max(0,     (x - kw + sx) / sx);
               int out_x_1 = min(out_w, (x      + sx) / sx);

               float val = 0;
               for (int out_y = out_y_0; out_y < out_y_1; ++out_y) {
                 for (int out_x = out_x_0; out_x < out_x_1; ++out_x) {
                   int offset = out_x + out_w * (out_y + out_h * c0);
                   val += gy[offset];
                 }
               }
               gx = val;
            ''',
            'max_pool_bwd')(x[0].reduced_view(),
                            self.h, self.w, y_h, y_w, self.kh, self.kw,
                            self.sy, self.sx, self.ph, self.pw,
                            gx)
        return gx,

    def create_pool_desc(self):
        return cudnn.create_pooling_descriptor(
            (self.kh, self.kw), (self.sy, self.sx), (self.ph, self.pw),
            libcudnn.CUDNN_POOLING_MAX)


def max_unpooling_2d(x, d_out, ksize, stride=None, pad=0, cover_all=True,
                   use_cudnn=True):
    """Spatial max unpooling function.

    This function acts similarly to :class:`~functions.Convolution2D`, but
    it computes the maximum of input spatial patch for each channel
    without any parameter instead of computing the inner products.

    Args:
        x (~chainer.Variable): Input variable.
        d_out (int, int): Size of output pixels
        ksize (int or (int, int)): Size of pooling window. ``ksize=k`` and
            ``ksize=(k, k)`` are equivalent.
        stride (int or (int, int) or None): Stride of pooling applications.
            ``stride=s`` and ``stride=(s, s)`` are equivalent. If None is
            specified, then it uses same stride as the pooling window size.
        pad (int or (int, int)): Spatial padding width for the input array.
            ``pad=p`` and ``pad=(p, p)`` are equivalent.
        cover_all (bool): If True, all spatial locations are pooled into some
            output pixels. It may make the output size larger.
        use_cudnn (bool): If True and CuDNN is enabled, then this function
            uses CuDNN as the core implementation. *** THIS VERSION DOES NOT SUPPORT THIS YET (TODO Tawara) ***

    Returns:
        ~chainer.Variable: Ouptut variable.

    """
    return MaxUnPooling2D( d_out, ksize, stride, pad, cover_all, use_cudnn)(x)


# Test unpooling function
def main():
    import numpy as np
    import chainer.functions as F
    from chainer import Variable, cuda
    import max_unpooling_2d

    rng = np.random.RandomState(42)
    # Generate random data
    n_data = 1
    height = 1
    width = 9
    in_channels = 1
    _X = rng.randn(n_data, in_channels, height, width).astype(np.float32)

    X = Variable(_X)
    X_gpu = Variable(cuda.to_gpu(_X))
    h = F.max_pooling_2d(X, (1, 3))
    
    maxpooling = F.MaxPooling2D((1,3))
    y=maxpooling(X)
    y_gpu=maxpooling(X_gpu)
    where = maxpooling.indexes

    # Test cpu mode
    max_unpooling = max_unpooling_2d.MaxUnPooling2D(X.data.shape[2:],(1,3))
    max_unpooling(y).data

    # Test GPU mode
    max_unpooling = max_unpooling_2d.MaxUnPooling2D(X_gpu.data.shape[2:],(1,3))
    max_unpooling(y_gpu).data

    max_unpooling.backward([y.data], [max_unpooling(y).data])
