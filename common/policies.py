import tensorflow as tf
from baselines.common import tf_util as U
from baselines.common.distributions import make_pdtype
from baselines.common import tf_util as U
from gym import spaces
import numpy as np

class MlpPolicy3(object):

    def __init__(self, X, sess, nact,  ac_space,  reuse=False, name="model"):
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

#not yet supported
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