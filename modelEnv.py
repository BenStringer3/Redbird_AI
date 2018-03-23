import gym
from gym.utils import seeding
import tensorflow as tf
from Redbird_AI.common.policies import RevConv
from baselines import logger
import numpy as np
import time
class Model2(object):
    def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                nsteps, max_grad_norm):
        name = "genEnv"
        img_size = 32
        tru_img_size = 32 #TODO: make robust
        img_shape =  [img_size, img_size, 1]
        tru_img_shape = [tru_img_size, tru_img_size, 1]
        sess = tf.get_default_session()
        np.random.seed(int(time.time()))

        with tf.device('/gpu:1'):
            X = tf.placeholder(tf.float32, (nbatch_act, ob_space.shape[0]), "X_act")
            act_model = policy(X, sess, img_size, ac_space, nbatch_act, 1, nlstm=256, reuse=False, name=name) # TODO remove hardcoded 10
            # X = tf.placeholder(tf.float32, (nbatch_train, ob_space.shape[0]), "X_train")
            # train_model = policy(X, sess, img_size, ac_space, nbatch_train, nsteps, nlstm=512, reuse=True, name=name)
            #TODO: watchout, I changed this training model to be like the act model
            train_model = act_model

            LR = tf.placeholder(tf.float32, [], name="LR2")

            OB_IMG_TRUE = tf.placeholder(
                tf.float32, [None] + tru_img_shape, name='OB_IMG_TRUE')
            if tru_img_shape[-1] == 3:
                ob_img_tru = tf.image.rgb_to_grayscale(OB_IMG_TRUE)
            else:
                ob_img_tru = OB_IMG_TRUE
            if tru_img_shape[0] != img_shape[0] or tru_img_shape[1] != img_shape[1]:
                ob_img_tru = tf.image.resize_images(ob_img_tru, [img_size, img_size], align_corners=True, method=3)

            loss = tf.reduce_mean(tf.square(train_model.ob_img - ob_img_tru))
            # loss = tf.reduce_mean(tf.square(act_model.ob_img - ob_img_tru))

            params = tf.trainable_variables(name)
            grads = tf.gradients(loss, params)
            if max_grad_norm is not None:
                general_grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
            grads = list(zip(general_grads, params))

            trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5, name="adam2")
            train_op = trainer.apply_gradients(grads)

            with tf.variable_scope("images"):
                for i in range(3): #TODO assumes there are at lest 3 nbatch_acts
                    tf.summary.image("ob_img_tru", [ob_img_tru[i, :, :, :]])
                    tf.summary.image("ob_img", [train_model.ob_img[i, :, :, :]])
                # tf.summary.image("ob_img", [act_model.ob_img[0, :, :, :]])
                for grad in general_grads:
                    if grad is not None:
                        tf.summary.histogram("grad sum", grad)
            summaries = tf.summary.merge_all(scope='images')# + tf.summary.merge_all(scope='genEnv')
            writer = tf.summary.FileWriter(logger.get_dir() + '/imgs')

        def train(ob, ob_img_true, lr):#, masks, states=None):
            grs = []
            for i in range(0, 40 , 4): #TODO remove hardcoded observation separation
                grs.append(np.array(ob)[:, i:i+2])
            # for gr in grs:
            #     td_map = {train_model.X:gr, train_model.S:states, train_model.M:masks}
            #     lstm, states = sess.run([train_model.LSTM, train_model.snew], td_map)

            #first
            masks = [True] * train_model.M.shape[0]
            a = sess.run(train_model.S, feed_dict={train_model.S: train_model.initial_state})
            td_map = {train_model.X: grs[0], train_model.M: masks}
            a = sess.run(train_model.state_op.op, td_map)

            #middle rmbas
            masks = [False] * train_model.M.shape[0]
            for gr in grs[1:-1]:
                td_map = {train_model.X:gr, train_model.M:masks}
                a = sess.run(train_model.state_op.op, td_map)

            #last rmba
            td_map = {train_model.X: grs[-1], train_model.M: masks}
            a = sess.run(train_model.lstm_op.op, td_map)

            ob_img_true = np.array(ob_img_true)
            # td_map = {train_model.IS_TRAINING:True, train_model.REV_CONV:lstm, OB_IMG_TRUE:ob_img_true, LR:lr }
            td_map = {train_model.IS_TRAINING: np.random.random_integers(0, 1) % 2 == 0, OB_IMG_TRUE: ob_img_true, LR: lr}
            # if states is not None:
            #     td_map[train_model.S] = states
            #     td_map[train_model.M] = masks
            train.counter += 1
            if train.counter % 200 ==  0: #TODO make optional argument
                stuff =  sess.run([loss,  summaries, train_op],td_map)[:-1]
                writer.add_summary(stuff[-1], global_step=logger.Logger.CURRENT.output_formats[0].step)
                writer.flush()
                return stuff[:-1]
            else:
                return sess.run([loss, train_op], td_map)[:-1]
        train.counter = -1

        self.train = train
        self.train_model = train_model #TODO uncommnet if needede
        self.act_model = act_model
        self.step = act_model.step
        # self.value = act_model.value
        self.initial_state = act_model.initial_state
        tf.global_variables_initializer().run(session=sess)

class ModelEnv(gym.Env):
    def __init__(self, ob_space, ac_space ):
        nenvs = 1 #TODO make vectorable
        nbatch_train = 1
        nsteps = 128
        max_grad_norm = 0.5
        from Redbird_AI.common.policies import LstmPolicy
        make_model = lambda: Model(policy=RevConv, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs,
                                   nbatch_train=nbatch_train,
                                   nsteps=nsteps,#, vf_coef=vf_coef,
                                   max_grad_norm=max_grad_norm)

        self.model = make_model()
        print("asdF")

    def reset(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step2(self, ob, ob_img_true):
        ob_img = self.model.step(ob)
        loss = self.model.train(ob_img, ob_img_true, 2.5e-4) #TODO: anneal lr
        return ob_img, loss



