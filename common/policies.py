import tensorflow as tf
from baselines.common import tf_util as U
from baselines.common.distributions import make_pdtype
from baselines.common import tf_util as U
from gym import spaces
import numpy as np
import math

class MlpPolicy3(object):

    def __init__(self, X, sess, nact, ac_space, nbatch, nsteps, nlstm=256, reuse=False, name="model"):
        with tf.variable_scope(name, reuse=reuse):
            ob = X
            l1 = tf.layers.dense(inputs=ob, units=512 , activation=tf.nn.tanh, name="l1")
            tf.summary.histogram(l1.name, l1)

            # l2 = tf.layers.dense(inputs=l1, units=512 * 2, activation=tf.nn.tanh, name="l2")

        # with tf.variable_scope("pi_layers", reuse=reuse):
            # logits branch
            l3 = tf.layers.dense(l1, 512, tf.nn.tanh, name="l3")
            l4 = tf.layers.dense(l3, 512, tf.nn.tanh, name="l4")
            pi = tf.layers.dense(l4, nact, activation=None, name="logits", kernel_initializer=U.normc_initializer(0.01))
            # logits = plain_dense(l5, pdtype.param_shape()[0], "logits", U.normc_initializer(0.01))
            # pi = plain_dense(l5, nact, "logits", U.normc_initializer(0.01))

        # with tf.variable_scope("vf_layers", reuse=reuse):
            # vpred branch
            # l3_v = tf.layers.dense(l1, 512, tf.nn.tanh, name="l3_v")
            # l4_v = tf.layers.dense(l3_v, 512, tf.nn.tanh, name="l4_v")
            # vf = plain_dense(l4_v, 1, "value", U.normc_initializer(1.0))[:, 0]
            vf = tf.layers.dense(l4, 1, name="value", kernel_initializer=U.normc_initializer(1.0))[:, 0]

        if isinstance(ac_space, spaces.Dict):
            self.pdtype0 = make_pdtype(ac_space.spaces["aav_pos"])
            self.pd0 = self.pdtype0.pdfromflat(pi)

            a0 = self.pd0.sample()
            neglogp0 = self.pd0.neglogp(a0)

            self.pdtype1 = make_pdtype(ac_space.spaces["exec"])
            self.pd1 = self.pdtype1.pdfromflat(pi)

            a1 = self.pd1.sample()
            neglogp1 = self.pd1.neglogp(a1)

            self.pdtype.sample_placeholder = tf.stack([self.pdtype0.sample_placeholder([None]), tf.cast(self.pdtype1.sample_placeholder([None]), tf.float32)], axis=0)

            def step(ob, *_args, **_kwargs):
                a, v, neglogp = sess.run([[a0, a1], vf, [neglogp0, neglogp1]], {X:ob})
                return a, v, self.initial_state, neglogp

            def value(ob, *_args, **_kwargs):
                return sess.run(vf, {X:ob})
        else:
            self.pdtype = make_pdtype(ac_space)
            self.pd = self.pdtype.pdfromflat(pi)

            a0 = self.pd.sample()
            neglogp0 = self.pd.neglogp(a0)
            self.initial_state = None

            def step(ob, *_args, **_kwargs):
                a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob})
                return a, v, self.initial_state, neglogp

            def value(ob, *_args, **_kwargs):
                return sess.run(vf, {X:ob})
        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

#pruning experiment
class MlpPolicy4(object):

    def _dense(self, x, size, name, weight_init=None, bias=True):
        with tf.name_scope(name):
            w = tf.contrib.model_pruning.apply_mask(tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=weight_init))
            ret = tf.matmul(x, w)
            if bias:
                b = tf.get_variable(name + "/b", [size], initializer=tf.zeros_initializer())
                return ret + b
            else:
                return ret

    def __init__(self, X, sess, nact, ac_space, reuse=False):
        with tf.variable_scope("model", reuse=reuse):
            ob = X
            # l1 = tf.layers.dense(inputs=ob, units=512 * 2, activation=tf.nn.tanh, name="l1")
            l1 = self._dense(ob,  1024, name="l1")

            tf.summary.histogram(l1.name, l1)

            l2 = tf.layers.dense(inputs=l1, units=512 * 2, activation=tf.nn.tanh, name="l2")

            # logits branch
            l3 = tf.layers.dense(l2, 512, tf.nn.tanh, name="l3")
            l4 = tf.layers.dense(l3, 512, tf.nn.tanh, name="l4")
            pi = tf.layers.dense(l4, nact, activation=None, name="logits",
                                 kernel_initializer=U.normc_initializer(
                                     np.mean(ac_space.high - ac_space.low) / nact))

            # vpred branch
            l3_v = tf.layers.dense(l2, 512, tf.nn.tanh, name="l3_v")
            l4_v = tf.layers.dense(l3_v, 512, tf.nn.tanh, name="l4_v")
            vf = tf.layers.dense(l4_v, 1, name="value", kernel_initializer=U.normc_initializer(1.0))[:, 0]

            self.pdtype = make_pdtype(ac_space)
            self.pd = self.pdtype.pdfromflat(pi)

            a0 = self.pd.sample()
            neglogp0 = self.pd.neglogp(a0)
            self.initial_state = None

            def step(ob, *_args, **_kwargs):
                a, v, neglogp = sess.run([a0, vf, neglogp0], {X: ob})
                return a, v, self.initial_state, neglogp

            def value(ob, *_args, **_kwargs):
                return sess.run(vf, {X: ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

#not yet supported (block sparse gpu kernels experiment)
class MlpPolicy5(object):

    def _sparse(self, inputs, units, name, weight_init=None, bias=True):
        from blocksparse.matmul import BlocksparseMatMul
        sparsity = np.random.randint(2, size=(inputs.shape[1] // units, inputs.shape[1] // units))



        with tf.name_scope(name):
            # Initialize the sparse matrix multiplication object
            bsmm = BlocksparseMatMul(sparsity, block_size=units, feature_axis=0)

            # Initialize block-sparse weights
            w = tf.get_variable("w", bsmm.w_shape, dtype=tf.float32)
            # w = tf.contrib.model_pruning.apply_mask(tf.get_variable(name + "/w", [inputs.get_shape()[1], units], initializer=weight_init))
            ret = bsmm(inputs, w)#tf.matmul(inputs, w)
            if bias:
                b = tf.get_variable(name + "/b", [units], initializer=tf.zeros_initializer())
                return ret + b
            else:
                return ret

    def __init__(self, X, sess, nact, ac_space, reuse=False):
        with tf.variable_scope("model", reuse=reuse):
            ob = X
            l1 = tf.nn.tanh(self._sparse(inputs=ob, units=16,  name="l1"))
            tf.summary.histogram(l1.name, l1)

            # l2 = tf.layers.dense(inputs=l1, units=512 * 2, activation=tf.nn.tanh, name="l2")

            # with tf.variable_scope("pi_layers", reuse=reuse):
            # logits branch
            l3 = tf.nn.tanh(self._sparse(inputs=l1, units=32,  name="l3"))#tf.layers.dense(l1, 512, tf.nn.tanh, name="l3")
            l4 = tf.nn.tanh(self._sparse(inputs=l3, units=32,  name="l2"))#tf.layers.dense(l3, 512, tf.nn.tanh, name="l4")
            pi = tf.layers.dense(l4, nact, activation=None, name="logits", kernel_initializer=U.normc_initializer(0.01))
            # logits = plain_dense(l5, pdtype.param_shape()[0], "logits", U.normc_initializer(0.01))
            # pi = plain_dense(l5, nact, "logits", U.normc_initializer(0.01))

            # with tf.variable_scope("vf_layers", reuse=reuse):
            # vpred branch
            l3_v = tf.nn.tanh(self._sparse(inputs=l1, units=32,  name="l1"))#tf.layers.dense(l1, 512, tf.nn.tanh, name="l3_v")
            l4_v = tf.nn.tanh(self._sparse(inputs=l3_v, units=32,  name="l1"))#tf.layers.dense(l3_v, 512, tf.nn.tanh, name="l4_v")
            # vf = plain_dense(l4_v, 1, "value", U.normc_initializer(1.0))[:, 0]
            vf = tf.layers.dense(l4_v, 1, name="value", kernel_initializer=U.normc_initializer(1.0))[:, 0]

        if isinstance(ac_space, spaces.Dict):
            self.pdtype0 = make_pdtype(ac_space.spaces["aav_pos"])
            self.pd0 = self.pdtype0.pdfromflat(pi)

            a0 = self.pd0.sample()
            neglogp0 = self.pd0.neglogp(a0)

            self.pdtype1 = make_pdtype(ac_space.spaces["exec"])
            self.pd1 = self.pdtype1.pdfromflat(pi)

            a1 = self.pd1.sample()
            neglogp1 = self.pd1.neglogp(a1)

            self.pdtype.sample_placeholder = tf.stack([self.pdtype0.sample_placeholder([None]),
                                                       tf.cast(self.pdtype1.sample_placeholder([None]),
                                                               tf.float32)], axis=0)

            def step(ob, *_args, **_kwargs):
                a, v, neglogp = sess.run([[a0, a1], vf, [neglogp0, neglogp1]], {X: ob})
                return a, v, self.initial_state, neglogp

            def value(ob, *_args, **_kwargs):
                return sess.run(vf, {X: ob})
        else:
            self.pdtype = make_pdtype(ac_space)
            self.pd = self.pdtype.pdfromflat(pi)

            a0 = self.pd.sample()
            neglogp0 = self.pd.neglogp(a0)
            self.initial_state = None

            def step(ob, *_args, **_kwargs):
                a, v, neglogp = sess.run([a0, vf, neglogp0], {X: ob})
                return a, v, self.initial_state, neglogp

            def value(ob, *_args, **_kwargs):
                return sess.run(vf, {X: ob})
        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

class LstmPolicy(object):
    # (self, X, sess, nact,  ac_space,  reuse=False, name="model"):
    def __init__(self, X, sess, nact, ac_space, nbatch, nsteps, nlstm=256, reuse=False, name="model"):
        from baselines.a2c.utils import batch_to_seq, seq_to_batch, lstm, fc
        nenv = nbatch // nsteps

        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm*2]) #states

        with tf.variable_scope(name, reuse=reuse):
            ob = X
            l1 = tf.layers.dense(inputs=ob, units=512, activation=tf.nn.tanh, name="l1")
            l2 = tf.layers.dense(inputs=l1, units=512, activation=tf.nn.tanh, name="l2")
            xs = batch_to_seq(l2, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            pi = tf.layers.dense(h5, nact, activation=None, name="logits", kernel_initializer=U.normc_initializer(0.01))
            vf = tf.layers.dense(h5, 1, name="value", activation=None, kernel_initializer=U.normc_initializer(1.0))[:, 0]

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pi)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((nenv, nlstm * 2), dtype=np.float32)

        def step(ob, state, mask):
            return sess.run([a0, vf, snew, neglogp0], {X:ob, S:state, M:mask})

        def value(ob, state, mask):
            return sess.run(vf, {X:ob, S:state, M:mask})

        self.X = X
        self.M = M
        self.S = S
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

class batch_norm(object):
    """Code modification of http://stackoverflow.com/a/33950177"""
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum

            self.name = name

    def __call__(self, x, train):
        return tf.contrib.layers.batch_norm(x, decay=self.momentum, updates_collections=None, epsilon=self.epsilon,
                                            center=True, scale=True, is_training=train, scope=self.name)

class RevConv(object):
    def __init__(self, X, sess, nact, #nact=image_size [64]
                 ac_space, nbatch, nsteps, nlstm=256, reuse=False, name="model"):
        # Currently, image size must be a (power of 2) and (8 or higher).
        assert(nact & (nact - 1) == 0 and nact >= 8)

        gf_dim = 64 #Dimension of gen filters in first conv layer. [64]
        log_size = int(math.log(nact) / math.log(2))
        g_bns = [
            batch_norm(name='g_bn{}'.format(i, )) for i in range(log_size)]
        IS_TRAINING = tf.placeholder(tf.bool, name='is_training')

        def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
            shape = input_.get_shape().as_list()

            with tf.variable_scope(scope or "Linear"):
                matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                         tf.random_normal_initializer(stddev=stddev))
                bias = tf.get_variable("bias", [output_size],
                                       initializer=tf.constant_initializer(bias_start))
                if with_w:
                    return tf.matmul(input_, matrix) + bias, matrix, bias
                else:
                    return tf.matmul(input_, matrix) + bias

        def conv2d_transpose(input_, output_shape,
                             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
                             name="conv2d_transpose", with_w=False):
            with tf.variable_scope(name):
                # filter : [height, width, output_channels, in_channels]
                w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                                    initializer=tf.random_normal_initializer(stddev=stddev))

                try:
                    deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                                    strides=[1, d_h, d_w, 1])

                # Support for verisons of TensorFlow before 0.7.0
                except AttributeError:
                    deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                            strides=[1, d_h, d_w, 1])

                biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
                # deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
                deconv = tf.nn.bias_add(deconv, biases)

                if with_w:
                    return deconv, w, biases
                else:
                    return deconv

        with tf.variable_scope(name, reuse=reuse):
            l1 = tf.layers.dense(inputs=X, units=512 , activation=tf.nn.tanh, name="l1")
            l2 = tf.layers.dense(inputs=l1, units=512 , activation=tf.nn.tanh, name="l2")
            z_, h0_w, h0_b = linear(l2, gf_dim * 8 * 4 * 4, 'g_h0_lin', with_w=True)
            # TODO: Nicer iteration pattern here. #readability
            hs = [None]
            hs[0] = tf.reshape(z_, [-1, 4, 4, gf_dim * 8])
            hs[0] = tf.nn.relu(g_bns[0](hs[0], IS_TRAINING))

            i = 1  # Iteration number.
            depth_mul = 8  # Depth decreases as spatial component increases.
            size = 8  # Size increases as depth decreases.

            while size < nact:
                hs.append(None)
                name = 'g_h{}'.format(i)
                hs[i], _, _ = conv2d_transpose(hs[i - 1],
                                               [nbatch, size, size, gf_dim * depth_mul], name=name,
                                               with_w=True)
                hs[i] = tf.nn.relu(g_bns[i](hs[i], IS_TRAINING))

                i += 1
                depth_mul //= 2
                size *= 2

            hs.append(None)
            name = 'g_h{}'.format(i)
            hs[i], _, _ = conv2d_transpose(hs[i - 1],
                                           [nbatch, size, size, 3], name=name, with_w=True)

            ob_img = tf.nn.tanh(hs[i])

        def step(ob, *_args, **_kwargs):
            return sess.run([ob_img], {X: ob, IS_TRAINING:False})


        self.X = X
        self.ob_img = ob_img
        self.step = step
        self.IS_TRAINING = IS_TRAINING

