import numpy as np
import tensorflow as tf
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm
from baselines.common.distributions import make_pdtype
from baselines.common import tf_util as U

def nature_cnn(unscaled_images):
    """
    CNN from Nature paper.
    """
    scaled_images = tf.cast(unscaled_images, tf.float32) / 255.
    activ = tf.nn.relu
    h = activ(conv(scaled_images, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2)))
    h2 = activ(conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2)))
    h3 = activ(conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2)))
    h3 = conv_to_fc(h3)
    return activ(fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2)))

class LnLstmPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, nlstm=256, reuse=False):
        nenv = nbatch // nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape) #obs
        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm*2]) #states
        with tf.variable_scope("model", reuse=reuse):
            h = nature_cnn(X)
            xs = batch_to_seq(h, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lnlstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            pi = fc(h5, 'pi', nact)
            vf = fc(h5, 'v', 1)

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pi)

        v0 = vf[:, 0]
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)

        def step(ob, state, mask):
            return sess.run([a0, v0, snew, neglogp0], {X:ob, S:state, M:mask})

        def value(ob, state, mask):
            return sess.run(v0, {X:ob, S:state, M:mask})

        self.X = X
        self.M = M
        self.S = S
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

class LstmPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, nlstm=256, reuse=False):
        nenv = nbatch // nsteps

        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape) #obs
        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm*2]) #states
        with tf.variable_scope("model", reuse=reuse):
            h = nature_cnn(X)
            xs = batch_to_seq(h, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            pi = fc(h5, 'pi', nact)
            vf = fc(h5, 'v', 1)

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pi)

        v0 = vf[:, 0]
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)

        def step(ob, state, mask):
            return sess.run([a0, v0, snew, neglogp0], {X:ob, S:state, M:mask})

        def value(ob, state, mask):
            return sess.run(v0, {X:ob, S:state, M:mask})

        self.X = X
        self.M = M
        self.S = S
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

class CnnPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False): #pylint: disable=W0613
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape) #obs
        with tf.variable_scope("model", reuse=reuse):
            h = nature_cnn(X)
            pi = fc(h, 'pi', nact, init_scale=0.01)
            vf = fc(h, 'v', 1)[:,0]

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

class MlpPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False): #pylint: disable=W0613
        ob_shape = (nbatch,) + ob_space.shape
        actdim = ac_space.shape[0]
        X = tf.placeholder(tf.float32, ob_shape, name='Ob') #obs
        with tf.variable_scope("model", reuse=reuse):
            activ = tf.tanh
            h1 = activ(fc(X, 'pi_fc1', nh=64, init_scale=np.sqrt(2)))
            h2 = activ(fc(h1, 'pi_fc2', nh=64, init_scale=np.sqrt(2)))
            pi = fc(h2, 'pi', actdim, init_scale=0.01)
            h1 = activ(fc(X, 'vf_fc1', nh=64, init_scale=np.sqrt(2)))
            h2 = activ(fc(h1, 'vf_fc2', nh=64, init_scale=np.sqrt(2)))
            vf = fc(h2, 'vf', 1)[:,0]
            logstd = tf.get_variable(name="logstd", shape=[1, actdim],
                initializer=tf.zeros_initializer())

        pdparam = tf.concat([pi, pi * 0.0 + logstd], axis=1)

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pdparam)

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

class MlpPolicy2(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False): #pylint: disable=W0613
        # ob_shape = ob_space.shape
        # actdim = ac_space.shape[0]
        # # X = tf.placeholder(tf.float32, shape= [None] + list(ob_space.shape) , name='Ob') #obs
        # X = tf.placeholder(tf.float32, ob_shape, name='Ob')  # obs
        # # pdtype = make_pdtype(ac_space)

        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc)
        nact = ac_space.n
        X = tf.placeholder(tf.float32, ob_shape)  # obs

        def plain_dense(x, size, name, weight_init=None, bias=True):
            w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=weight_init)
            ret = tf.matmul(x, w)
            if bias:
                b = tf.get_variable(name + "/b", [size], initializer=tf.zeros_initializer())
                return ret + b
            else:
                return ret

        with tf.variable_scope("model", reuse=reuse):
            l1 = tf.layers.dense(inputs=X, units=512 * 4, activation=tf.nn.tanh, name="l1")
            l2 = tf.layers.dense(inputs=l1, units=512 * 3, activation=tf.nn.tanh, name="l2")

            # logits branch
            l3 = tf.layers.dense(l2, 512 * 2, tf.nn.tanh, name="l3")
            l4 = tf.layers.dense(l3, 64 * 4, tf.nn.tanh, name="l4")
            l5 = tf.layers.dense(l4, 64 * 4, tf.nn.tanh, name="l5")
            # logits = tf.layers.dense(l5, pdtype.param_shape()[0], name="logits", kernel_initializer=U.normc_initializer(0.01))
            # logits = plain_dense(l5, pdtype.param_shape()[0], "logits", U.normc_initializer(0.01))
            logits = plain_dense(l5, nact, "logits", U.normc_initializer(0.01))

            # vpred branch
            l3_v = tf.layers.dense(l2, 512 * 2, tf.nn.tanh, name="l3_v")
            l4_v = tf.layers.dense(l3_v, 64 * 4, tf.nn.tanh, name="l4_v")
            l5_v = tf.layers.dense(l4_v, 64 * 4, tf.nn.tanh, name="l5_v")
            vf = plain_dense(l5_v, 1, "value", U.normc_initializer(1.0))[:, 0]

            # activ = tf.tanh
            # h1 = activ(fc(X, 'pi_fc1', nh=64, init_scale=np.sqrt(2)))
            # h2 = activ(fc(h1, 'pi_fc2', nh=64, init_scale=np.sqrt(2)))
            # pi = fc(h2, 'pi', pdtype.param_shape()[0], init_scale=0.01)
            # h1 = activ(fc(X, 'vf_fc1', nh=64, init_scale=np.sqrt(2)))
            # h2 = activ(fc(h1, 'vf_fc2', nh=64, init_scale=np.sqrt(2)))
            # vf = fc(h2, 'vf', 1)[:,0]
            # logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]],
            logstd = tf.get_variable(name="logstd", shape=[1, actdim],
                initializer=tf.zeros_initializer())

        pdparam = tf.concat([logits, logits * 0.0 + logstd], axis=1)

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pdparam)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob})
            return a[0], v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})



        self.X = X
        self.pi = logits
        self.vf = vf
        self.step = step
        self.value = value

class MlpPolicy3(object):

    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False): #pylint: disable=W0613
        # nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, ob_space.shape[0])
        nact = np.sum(ac_space.nvec)
        X = tf.placeholder(tf.float32, ob_shape) #obs

        def plain_dense(x, size, name, weight_init=None, bias=True):
            w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=weight_init)
            ret = tf.matmul(x, w)
            if bias:
                b = tf.get_variable(name + "/b", [size], initializer=tf.zeros_initializer())
                return ret + b
            else:
                return ret

        with tf.variable_scope("general_layers", reuse=reuse):
            l1 = tf.layers.dense(inputs=X, units=512 * 2, activation=tf.nn.tanh, name="l1")
            l2 = tf.layers.dense(inputs=l1, units=512 * 2, activation=tf.nn.tanh, name="l2")

        with tf.variable_scope("pi_layers", reuse=reuse):
            # logits branch
            l3 = tf.layers.dense(l2, 512, tf.nn.tanh, name="l3")
            l4 = tf.layers.dense(l3, 512, tf.nn.tanh, name="l4")
            pi = tf.layers.dense(l4, nact, activation=None, name="logits", kernel_initializer=U.normc_initializer(0.01))
            # logits = plain_dense(l5, pdtype.param_shape()[0], "logits", U.normc_initializer(0.01))
            # pi = plain_dense(l5, nact, "logits", U.normc_initializer(0.01))

        with tf.variable_scope("vf_layers", reuse=reuse):
            # vpred branch
            l3_v = tf.layers.dense(l2, 512, tf.nn.tanh, name="l3_v")
            l4_v = tf.layers.dense(l3_v, 512, tf.nn.tanh, name="l4_v")
            # vf = plain_dense(l4_v, 1, "value", U.normc_initializer(1.0))[:, 0]
            vf = tf.layers.dense(l4_v, 1, name="value", kernel_initializer=U.normc_initializer(1.0))[:, 0]

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