import tensorflow as tf
import gym
import baselines.common.tf_util as U
from baselines.common.distributions import make_pdtype


class RedbirdPolicy(object):
    def __init__(self, name, ob_space, ac_space, kind):
        with tf.variable_scope(name):
            self._init(ob_space, ac_space, kind)
            self.scope = tf.get_variable_scope().name

    def plain_dense(self, x, size, name, weight_init=None, bias=True):
        w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=weight_init)
        ret = tf.matmul(x, w)
        if bias:
            b = tf.get_variable(name + "/b", [size], initializer=tf.zeros_initializer())
            return ret + b
        else:
            return ret

    def _init(self, ob_space, ac_space, kind):
        assert isinstance(ob_space, gym.spaces.Box)

        #setup probability distribution based on action space
        self.pdtype = pdtype = make_pdtype(ac_space)

        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[None] + list(ob_space.shape))

        if kind == 'denseSplit': #ben's network structure
            l1 = tf.layers.dense(inputs=ob, units=512 * 4, activation=tf.nn.tanh, name="l1")
            l2 = tf.layers.dense(inputs=l1, units=512 * 3, activation=tf.nn.tanh, name="l2")

            # logits branch
            l3 = tf.layers.dense(l2, 512 * 2, tf.nn.tanh, name="l3")
            l4 = tf.layers.dense(l3, 64 * 4, tf.nn.tanh, name="l4")
            l5 = tf.layers.dense(l4, 64 * 4, tf.nn.tanh, name="l5")
            # logits = tf.layers.dense(l5, pdtype.param_shape()[0], name="logits", kernel_initializer=U.normc_initializer(0.01))
            logits = self.plain_dense(l5, pdtype.param_shape()[0], "logits", U.normc_initializer(0.01))
            self.pd = pdtype.pdfromflat(logits)

            #vpred branch
            l3_v = tf.layers.dense(l2, 512 * 2, tf.nn.tanh, name="l3_v")
            l4_v = tf.layers.dense(l3_v, 64 * 4, tf.nn.tanh, name="l4_v")
            l5_v = tf.layers.dense(l4_v, 64 * 4, tf.nn.tanh, name="l5_v")
            self.vpred = self.plain_dense(l5_v, 1, "value", U.normc_initializer(1.0))[:, 0]
            # self.vpred = tf.layers.dense(l5, 1, name="value", kernel_initializer=U.normc_initializer(1.0))[:, 0]
        elif kind == 'dense':
            l1 = tf.layers.dense(inputs=ob, units=512 * 4, activation=tf.nn.tanh, name="l1")
            l2 = tf.layers.dense(inputs=l1, units=512 * 3, activation=tf.nn.tanh, name="l2")
            l3 = tf.layers.dense(l2, 512 * 2, tf.nn.tanh, name="l3")
            l4 = tf.layers.dense(l3, 64 * 4, tf.nn.tanh, name="l4")
            l5 = tf.layers.dense(l4, 64 * 4, tf.nn.tanh, name="l5")
            # logits = tf.layers.dense(l5, pdtype.param_shape()[0], name="logits", kernel_initializer=U.normc_initializer(0.01))
            logits = self.plain_dense(l5, pdtype.param_shape()[0], "logits", U.normc_initializer(0.01))
            self.pd = pdtype.pdfromflat(logits)
            self.vpred = self.plain_dense(l5, 1, "value", U.normc_initializer(1.0))[:, 0]
        else:
            raise NotImplementedError

        # output of network, the actions
        ac = self.pd.sample()

        # what does this do? -ben
        stochastic = tf.placeholder(dtype=tf.bool, shape=())

        self._act = U.function([stochastic, ob], [ac, self.vpred])

    def act(self, stochastic, ob):
        ac1, vpred1 =  self._act(stochastic, ob[None])
        return ac1[0], vpred1#[0] #ben
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_initial_state(self):
        return []