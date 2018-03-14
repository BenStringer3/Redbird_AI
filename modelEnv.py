import gym
from gym.utils import seeding
import tensorflow as tf
from Redbird_AI.common.policies import RevConv

class Model(object):
    def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                nsteps, vf_coef, max_grad_norm):
        img_size = 64
        img_shape =  [img_size, img_size, 1]
        sess = tf.get_default_session()

        X = tf.placeholder(tf.float32, (nbatch_act, ob_space.shape[0]), "X")
        act_model = policy(X, sess, img_size, ac_space, nbatch_act, 1, nlstm=512, reuse=False, name="genEnv")
        X = tf.placeholder(tf.float32, (nbatch_train, ob_space.shape[0]), "X_1")
        train_model = policy(X, sess, img_size, ac_space, nbatch_train, nsteps, nlstm=512, reuse=True, name="genEnv")

        LR = tf.placeholder(tf.float32, [], name="LR2")
        OB_IMG = tf.placeholder(
            tf.float32, [None] + img_shape, name='OB_IMG')
        OB_IMG_TRUE = tf.placeholder(
            tf.float32, [None] + img_shape, name='OB_IMG_TRUE')
        loss = tf.reduce_sum(tf.pow(OB_IMG - OB_IMG_TRUE, 2.))
        params = tf.trainable_variables("genEnv")
        grads = tf.gradients(loss, params)

        trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5, name="adam2")
        _train = trainer.apply_gradients(grads)

        def train(ob_img, ob_img_true, lr):#, masks, states=None):
            td_map = {train_model.IS_TRAINING:True, OB_IMG:ob_img, OB_IMG_TRUE:ob_img_true, LR:lr }
            # if states is not None:
            #     td_map[train_model.S] = states
            #     td_map[train_model.M] = masks
            stuff =  sess.run(
                [loss, _train],#, summaries],
                td_map
            )[:-1]
            # logger.Logger.CURRENT.writer.add_summary(stuff[-1], global_step=logger.Logger.CURRENT.step)
            return stuff #[:-1]

        self.train = train
        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value
        self.initial_state = act_model.initial_state
        tf.global_variables_initializer().run(session=sess)

class modelEnv(gym.Env):
    def __init__(self):
        make_model = lambda: Model(policy=RevConv, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs,
                                   nbatch_train=nbatch_train,
                                   nsteps=nsteps, vf_coef=vf_coef,
                                   max_grad_norm=max_grad_norm, gpu=gpu)

        self.model = make_model()

    def reset(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step2(self, ob, ob_img_true):
        ob_img = self.model.step(ob)
        loss = self.model.train(ob_img, ob_img_true, 2.5e-4) #TODO: anneal lr
        return ob_img, loss



