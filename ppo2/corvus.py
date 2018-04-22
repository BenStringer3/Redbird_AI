import numpy as np
import tensorflow as tf
from baselines import logger
import gym
from gym import spaces
from Redbird_AI.modelEnv import Model2
from Redbird_AI.common.policies import LSTM_GR_Viewer

class Model(object):
    def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                nsteps, vf_coef, max_grad_norm, gpu=0):
        sess = tf.get_default_session()

        with tf.device('/device:GPU:'+ str(gpu)):
            if isinstance(ac_space, spaces.Box):
                nact = np.sum(ac_space.shape) *2
            else:
                nact = np.sum(ac_space.nvec)

            self.aav_pos_act = tf.placeholder(tf.float32, (nbatch_act, 2), "aav_pos_act")
            aav_pos_train = tf.placeholder(tf.float32, (nbatch_train, 2), "aav_pos_train")
            self.X = tf.placeholder(tf.float32, (nbatch_act, 4), "X")
            X = tf.placeholder(tf.float32, (nbatch_train, 4), "X_1")
            lstm_viewer_train = LSTM_GR_Viewer(X, sess=None, nact=None, ac_space=None,
                                         nbatch=nbatch_train, nsteps=nsteps, nlstm=66, reuse=False,
                                         name='GR_viewer')
            lstm_viewer_act = LSTM_GR_Viewer(self.X, sess=None, nact=None, ac_space=None,
                                         nbatch=nbatch_act, nsteps=1, nlstm=66, reuse=True,
                                         name='GR_viewer')
            act_model = policy( tf.concat((lstm_viewer_act.Y, self.aav_pos_act),1), sess, nact, ac_space, nbatch_act, 1, nlstm=32, reuse=False) # (sess, ob_space, ac_space, [nbatch_act], 1, reuse=False)

            train_model = policy( tf.concat((lstm_viewer_train.Y, aav_pos_train),1), sess, nact, ac_space, nbatch_train, nsteps,nlstm=32, reuse=True)#(sess, ob_space, ac_space, [nbatch_train], nsteps, reuse=True)

            A = train_model.pdtype.sample_placeholder([None])
            ADV = tf.placeholder(tf.float32, [None])
            R = tf.placeholder(tf.float32, [None])
            OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
            OLDVPRED = tf.placeholder(tf.float32, [None])
            LR = tf.placeholder(tf.float32, [], name="LR")
            ENT_COEFF = tf.placeholder(tf.float32, [], name="ENT_COEFF")
            CLIPRANGE = tf.placeholder(tf.float32, [], name="CLIPRANGE")

            neglogpac = train_model.pd.neglogp(A)
            entropy = tf.reduce_mean(train_model.pd.entropy(),name="entropy")

            vpred = train_model.vf
            vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
            vf_losses1 = tf.square(vpred - R)
            vf_losses2 = tf.square(vpredclipped - R)
            vf_loss_ = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
            ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
            pg_losses = -ADV * ratio
            pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
            pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
            approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
            clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))
            pi_loss = pg_loss - entropy * ENT_COEFF
            vf_loss = vf_loss_ * vf_coef
            general_loss = pi_loss + vf_loss

            params = tf.trainable_variables("model")
            grads = tf.gradients(general_loss, params)

            if max_grad_norm is not None:
                general_grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
            grads = list(zip(general_grads, params))

            trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5, name="adam")
            _train = trainer.apply_gradients(grads)

        #debugging/profiling
        summaries = tf.summary.merge_all(scope="model")
        writer = tf.summary.FileWriter(logger.get_dir() + '/mdl_sums')
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        # Create options to profile the time and memory information.
        builder = tf.profiler.ProfileOptionBuilder
        opts = builder(builder.time_and_memory()).order_by('micros').build()



        def train(lr, cliprange, obs, returns, masks, actions, values, neglogpacs, ent_coeff, states=None):

            nenvs = obs.shape[0]
            grs = []

            for j in range(0, 40, 4):
                rmba = np.array(obs)[:, j:j + 4]
                grs.append(rmba)

            # first-----------------------------------------
            # forget all prior GR's in GR-viewer
            masks1 = [True] * lstm_viewer_train.M.shape[0]
            states1 = lstm_viewer_train.initial_state

            # all rmbas but the last
            for gr in grs[:-1]:
                td_map = {lstm_viewer_train.M: masks1, lstm_viewer_train.S: states1}
                masks1 = [False] * lstm_viewer_train.M.shape[0]
                td_map[lstm_viewer_train.X] = gr
                _, states1 = sess.run([lstm_viewer_train.Y, lstm_viewer_train.snew], td_map)  # train_model.lstm_op.op,


            # start of regular train
            advs = returns - values
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            td_map = {lstm_viewer_train.X:grs[-1], lstm_viewer_train.S:states1, lstm_viewer_train.M:masks1,
                      aav_pos_train:obs[:,40:42],
                      A: actions, ADV: advs, R: returns, LR: lr,
                      CLIPRANGE: cliprange, OLDNEGLOGPAC: neglogpacs, OLDVPRED: values, ENT_COEFF: ent_coeff}
            # td_map = {train_model.X:obs, A:actions, ADV:advs, R:returns, LR:lr,
            #         CLIPRANGE:cliprange, OLDNEGLOGPAC:neglogpacs, OLDVPRED:values, ENT_COEFF:ent_coeff}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            train.counter += 1
            if train.counter % 200 == 0: # TODO make optional arg
                # Create a profiling context, set constructor argument `trace_steps`,
                # `dump_steps` to empty for explicit control.
                # with tf.contrib.tfprof.ProfileContext('/tmp/train_dir',
                #                                       trace_steps=[],
                #                                       dump_steps=[]) as pctx:
                #     # with sess.as_default():
                #     sess = tf.get_default_session()
                #     # Enable tracing for next session.run.
                #     pctx.trace_next_step()
                #     # Dump the profile to '/tmp/train_dir' after the step.
                #     pctx.dump_next_step()
                #     stuff = sess.run(
                #         [pg_loss, vf_loss, entropy, approxkl, clipfrac, general_loss, summaries, _train],
                #         td_map  # , options=run_options, run_metadata=run_metadata
                #     )[:-1]
                #     pctx.profiler.profile_operations(options=opts)
                stuff = sess.run(
                    [pg_loss, vf_loss, entropy, approxkl, clipfrac, general_loss, summaries, _train],
                    td_map #, options=run_options, run_metadata=run_metadata
                )[:-1]
                writer.add_summary(stuff[-1], global_step=logger.Logger.CURRENT.output_formats[0].step)
                writer.flush()
                return stuff[:-1]
            else:
                # sess = tf.get_default_session() #TODO remove if profiling doesn't work out
                return sess.run(
                    [pg_loss, vf_loss, entropy, approxkl, clipfrac, general_loss, _train],
                    td_map
                )[:-1]
        train.counter = -1
        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac', 'total_loss']
        self.S = act_model.S
        self.M = act_model.M
        self.A = act_model.A
        self.vf = act_model.vf
        self.snew = act_model.snew
        self.neglogp0 = act_model.neglogp0
        self.train = train
        self.train_model = train_model
        self.act_model = act_model
        self.lstm_viewer_act = lstm_viewer_act
        self.step = act_model.step
        self.value = act_model.value
        self.initial_state = act_model.initial_state
        # self.save = save
        # self.load = load
        tf.global_variables_initializer().run(session=sess) #pylint: disable=E1101

class Corvus(object):
    def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                nsteps, max_grad_norm, gpu, nenvs, vf_coef):
        self.numGRs = 10
        self.numParams= 4
        make_policy_model = lambda: Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs,
                                          nbatch_train=nbatch_train,
                                          nsteps=nsteps, vf_coef=vf_coef,
                                          max_grad_norm=max_grad_norm, gpu=gpu)

        make_generative_environment_model = lambda: Model2(policy=None,
                                                           ob_space=gym.spaces.Box(np.array([0, 0]),
                                                                                   np.array([20.0, 20.0]),
                                                                                   dtype=np.float32),
                                                           ac_space=gym.spaces.Box(0, 255, [64, 64]),
                                                           nbatch_act=nenvs, nbatch_train=nbatch_train,
                                                           nsteps=10,  # TODO remove hardcode
                                                           max_grad_norm=max_grad_norm)

        self.policy_model = make_policy_model()
        self.genEnv_model = make_generative_environment_model()
        self.genEnv_snew = self.genEnv_model.initial_state
        self.loss_names = self.policy_model.loss_names
        self.counter = 0
        self.sess = tf.get_default_session()

    def _feedGRs(self, genEnv_obs):
        grs = []
        for j in range(0, self.numParams * self.numGRs, self.numParams):
            rmba = np.array(genEnv_obs)[:, j:j + 2]
            grs.append(rmba)

        # first-----------------------------------------
        # forget all prior GR's in GR-viewer
        masks = [True] * self.genEnv_model.lstm_viewer.M.shape[0]
        states = self.genEnv_model.lstm_viewer.initial_state

        # all rmbas but the last
        for gr in grs[:-1]:
            td_map = {self.genEnv_model.lstm_viewer.M: masks, self.genEnv_model.lstm_viewer.S: states}
            masks = [False] * self.genEnv_model.lstm_viewer.M.shape[0]
            td_map[self.genEnv_model.lstm_viewer.X] = gr
            _, states = self.sess.run([self.genEnv_model.lstm_viewer.Y, self.genEnv_model.lstm_viewer.snew], td_map)
        return grs[-1], states

    def step(self, policy_obs, policy_states,  policy_masks=None):
        td_map = {self.policy_model.X: policy_obs, self.policy_model.S: policy_states, self.policy_model.M: policy_masks}
        return self.sess.run([self.policy_model.A, self.policy_model.vf, self.policy_model.snew, self.policy_model.neglogp0], td_map)

    def step3(self, policy_obs, policy_states,  policy_masks=None):
        nenvs = policy_obs.shape[0]
        grs = []

        for j in range(0, 40, 4):
            rmba = np.array(policy_obs)[:, j:j + 4]
            grs.append(rmba)

        # first-----------------------------------------
        # forget all prior GR's in GR-viewer
        masks1 = [True] * self.policy_model.lstm_viewer_act.M.shape[0]
        states1 = self.policy_model.lstm_viewer_act.initial_state

        # all rmbas but the last
        for gr in grs[:-1]:
            td_map = {self.policy_model.lstm_viewer_act.M: masks1, self.policy_model.lstm_viewer_act.S: states1}
            masks1 = [False] * self.policy_model.lstm_viewer_act.M.shape[0]
            td_map[self.policy_model.lstm_viewer_act.X] = gr
            _, states1 = self.sess.run([self.policy_model.lstm_viewer_act.Y, self.policy_model.lstm_viewer_act.snew], td_map)  # train_model.lstm_op.op,
        td_map = {self.policy_model.lstm_viewer_act.X: grs[-1], self.policy_model.lstm_viewer_act.S:states1, self.policy_model.lstm_viewer_act.M:masks1,
                  self.policy_model.aav_pos_act: policy_obs[:,40:42],
                  self.policy_model.S: policy_states,
                  self.policy_model.M: policy_masks}
        return self.sess.run(
            [self.policy_model.A, self.policy_model.vf, self.policy_model.snew, self.policy_model.neglogp0], td_map)

    def step2(self, lr, imgs, policy_obs,  genEnv_obs, policy_states, genEnv_states, policy_masks=None, genEnv_masks=None):

        lastGR, feedStates = self._feedGRs(genEnv_obs)

        # last GR
        td_map = {self.genEnv_model.rev_conv.IS_TRAINING: True,
                  self.genEnv_model.lstm_viewer.X: lastGR,
                  self.genEnv_model.lstm_viewer.M: [False] * self.genEnv_model.lstm_viewer.M.shape[0],
                  self.genEnv_model.lstm_viewer.S: feedStates,
                  self.genEnv_model.LR: lr*4,
                  self.genEnv_model.OB_IMG_TRUE: imgs,
                  self.policy_model.X: policy_obs}
        if genEnv_masks is not None:
            td_map[self.genEnv_model.inGame_memory.S] = genEnv_states
            td_map[self.genEnv_model.inGame_memory.M] = genEnv_masks
        if policy_masks is not None:
            td_map[self.policy_model.S] = policy_states
            td_map[self.policy_model.M] = policy_masks

        tensors = [self.policy_model.A, self.policy_model.vf, self.policy_model.snew, self.policy_model.neglogp0,
                        self.genEnv_model.loss, self.genEnv_model.inGame_memory.snew]


        self.counter += 1
        if self.counter % max(12+1, 0) ==  0:
            ac, vf, pol_snew, neglogp0, genEnv_loss, genEnv_snew, summ = self.sess.run([
                        self.policy_model.A, self.policy_model.vf, self.policy_model.snew, self.policy_model.neglogp0,
                        self.genEnv_model.loss, self.genEnv_model.inGame_memory.snew, self.genEnv_model.summaries,
                        self.genEnv_model.train_op], td_map)[:-1]
            self.genEnv_model.writer.add_summary(summ, global_step=logger.Logger.CURRENT.output_formats[0].step)
            return ac, vf, pol_snew, neglogp0, genEnv_loss, genEnv_snew
        else:
            ac, vf, pol_snew, neglogp0, genEnv_loss, genEnv_snew = self.sess.run([
                self.policy_model.A, self.policy_model.vf, self.policy_model.snew, self.policy_model.neglogp0,
                self.genEnv_model.loss, self.genEnv_model.inGame_memory.snew,
                self.genEnv_model.train_op, self.genEnv_model.enque_op], td_map)[:-2]
            return ac, vf, pol_snew, neglogp0, genEnv_loss, genEnv_snew

    def value(self, ob, state, mask):
        # return self.sess.run(self.policy_model.vf, {self.policy_model.X:ob, self.policy_model.S: state, self.policy_model.M:mask})
        nenvs = ob.shape[0]
        grs = []

        for j in range(0, 40, 4):
            rmba = np.array(ob)[:, j:j + 4]
            grs.append(rmba)

        # first-----------------------------------------
        # forget all prior GR's in GR-viewer
        masks1 = [True] * self.policy_model.lstm_viewer_act.M.shape[0]
        states1 = self.policy_model.lstm_viewer_act.initial_state

        # all rmbas but the last
        for gr in grs[:-1]:
            td_map = {self.policy_model.lstm_viewer_act.M: masks1, self.policy_model.lstm_viewer_act.S: states1}
            masks1 = [False] * self.policy_model.lstm_viewer_act.M.shape[0]
            td_map[self.policy_model.lstm_viewer_act.X] = gr
            _, states1 = self.sess.run([self.policy_model.lstm_viewer_act.Y, self.policy_model.lstm_viewer_act.snew], td_map)  # train_model.lstm_op.op,
        td_map = {self.policy_model.lstm_viewer_act.X: grs[-1], self.policy_model.lstm_viewer_act.S:states1, self.policy_model.lstm_viewer_act.M:masks1,
                  self.policy_model.aav_pos_act:ob[:,40:42],
                  self.policy_model.S: state,
                  self.policy_model.M: mask}
        return self.sess.run(
            self.policy_model.vf, td_map)

    def train(self,lr, cliprange,  obs,returns, masks, actions, values,
              neglogpacs, imgs, ent_coeff, policy_states=None):
        genEnv_loss, self.genEnv_snew = self.genEnv_model.train(obs, imgs,  masks, 4 * lr, self.genEnv_snew)

        policy_losses = self.policy_model.train(lr, cliprange, obs, returns, masks, actions, values, neglogpacs,
                                       ent_coeff, policy_states)

        return (*policy_losses, genEnv_loss)

    def train2(self, lr, cliprange,  policy_obs,  returns, policy_masks, actions,
                  values,
                  neglogpacs, ent_coeff, policy_states=None):
        # genEnv_loss, genEnv_snew = self.genEnv_model.train(genEnv_obs, imgs, 4*lr, genEnv_masks, genEnv_states)

        return  self.policy_model.train(lr, cliprange, policy_obs, returns, policy_masks, actions, values, neglogpacs, ent_coeff, policy_states)

        # return genEnv_loss, genEnv_snew, policy_losses