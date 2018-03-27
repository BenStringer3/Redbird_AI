import gym
from gym.utils import seeding
import tensorflow as tf
from Redbird_AI.common.policies import RevConv2, LSTM_GR_Viewer
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
            # X = tf.placeholder(tf.float32, (nbatch_act, ob_space.shape[0]), "X_act")
            # act_model = policy(X, sess, img_size, ac_space, nbatch_act, 1, nlstm=512, reuse=False, name=name) # TODO remove hardcoded 10
            # # X = tf.placeholder(tf.float32, (nbatch_train, ob_space.shape[0]), "X_train")
            # # train_model = policy(X, sess, img_size, ac_space, nbatch_train, nsteps, nlstm=512, reuse=True, name=name)
            # #TODO: watchout, I changed this training model to be like the act model
            # train_model = act_model

            try:
                X = tf.placeholder(tf.float32, (nbatch_act, ob_space.shape[0]), "X_act")
            except TypeError:
                X = tf.placeholder(tf.float32, (nbatch_act, ob_space.spaces[0].shape[0]), "X_act")
            lstm_viewer = LSTM_GR_Viewer(X, sess, img_size, ac_space, nbatch_act, 1, nlstm=512, reuse=False, name=name)# + '_LSTM_GR_Viewer')
            rev_conv = RevConv2(lstm_viewer.Y, sess, img_size, ac_space, nbatch_act, 1, nlstm=512, reuse=False, name=name)# + 'Rev_Conv')


            LR = tf.placeholder(tf.float32, [], name="LR2")

            OB_IMG_TRUE = tf.placeholder(
                tf.uint8, [nbatch_act] + tru_img_shape, name='OB_IMG_TRUE')
            if tru_img_shape[-1] == 3:
                ob_img_tru = tf.image.rgb_to_grayscale(OB_IMG_TRUE)
            else:
                ob_img_tru = OB_IMG_TRUE
            if tru_img_shape[0] != img_shape[0] or tru_img_shape[1] != img_shape[1]:
                ob_img_tru = tf.image.resize_images(ob_img_tru, [img_size, img_size], align_corners=True, method=3)

            rand = tf.random_normal((), mean=3.0, stddev=2.)
            ob_img_tru = tf.cast(ob_img_tru, tf.float32)
            ob_img_tru_w_noise = tf.clip_by_value(ob_img_tru + tf.random_normal(ob_img_tru.shape, mean=rand, stddev=3.), 0.0, 255.0)

            loss = tf.reduce_mean(tf.square(rev_conv.Y - ob_img_tru))
            loss_w_noise = tf.reduce_mean(tf.square(rev_conv.Y - ob_img_tru_w_noise))
            # loss = tf.reduce_mean(tf.square(act_model.ob_img - ob_img_tru))

            params = tf.trainable_variables(name)# + tf.trainable_variables(name + 'Rev_Conv')
            grads = tf.gradients(loss_w_noise, params)
            if max_grad_norm is not None:
                general_grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
            grads = list(zip(general_grads, params))

            trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5, name="adam2")
            train_op = trainer.apply_gradients(grads)

            with tf.variable_scope("images"):
                for i in range(3): #TODO assumes there are at lest 3 nbatch_acts
                    tf.summary.image("ob_img_tru", [ob_img_tru_w_noise[i, :, :, :]])
                    tf.summary.image("ob_img", [rev_conv.Y[i, :, :, :]])
                # tf.summary.image("ob_img", [act_model.ob_img[0, :, :, :]])
                # for grad in general_grads:
                #     if grad is not None:
                #         tf.summary.histogram("grad sum", grad)
            summaries = tf.summary.merge_all(scope='images')# + tf.summary.merge_all(scope='genEnv')
            writer = tf.summary.FileWriter(logger.get_dir() + '/imgs')

        def train(ob, ob_img_true, lr):#, masks, states=None):
            nenvs = ob.shape[0]
            grs = np.ones([nenvs, 10, 2]) * np.nan
            iter = [0] * nenvs
            for i in range(nenvs):
                for j in range(0, 40 , 4):
                    rmba = np.array(ob)[i, j:j+2]
                    grs[i,iter[i],:] = rmba
                    if not np.isnan(rmba).any():
                        iter[i] += 1

            #first-----------------------------------------
            # forget all prior GR's in GR-viewer
            masks1 = [True] * lstm_viewer.M.shape[0]
            #start GR-viewer with a blank memory
            sess.run(lstm_viewer.S, feed_dict={lstm_viewer.S: lstm_viewer.initial_state})
            memories = np.zeros(lstm_viewer.S.shape)

            #all rmbas but the last
            for i in range(10 - 1):
                gr = grs[:, i, :]
                td_map = {lstm_viewer.M:masks1}
                masks1 = np.prod(np.isnan(gr), axis=1)
                where_are_NaNs = np.isnan(gr)
                gr[where_are_NaNs] = 0
                td_map[lstm_viewer.X ] = gr
                _, states1 = sess.run([lstm_viewer.Y, lstm_viewer.state_op],  td_map) # train_model.lstm_op.op,
                for j in range(nenvs):
                    if not masks1[j]:
                        memories[j] = states1[j]

            ob_img_true = np.array(ob_img_true)

            #last GR
            last_gr = np.array([grs[i, iter[i]-1, :] for i in range(nenvs)])
            masks1 = np.prod(np.isnan(last_gr), axis=1)
            where_are_NaNs = np.isnan(grs)
            grs[where_are_NaNs] = 0
            last_gr = np.array([grs[i, iter[i] - 1, :] for i in range(nenvs)])
            td_map = {rev_conv.IS_TRAINING: True, lstm_viewer.X:last_gr,
                      OB_IMG_TRUE:ob_img_true, lstm_viewer.M:masks1, LR: lr,
                      lstm_viewer.S:memories}


            train.counter += 1
            if train.counter % 195 ==  0: #TODO make optional argument
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
        # self.step = act_model.step
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



