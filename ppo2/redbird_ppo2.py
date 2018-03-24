import os
import time
import joblib
import numpy as np
import os.path as osp
import tensorflow as tf
from baselines import logger
from collections import deque
from baselines.common import explained_variance
from Redbird_AI.common.cmd_util import load_model, save_model
from gym import spaces

class Model(object):
    def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                nsteps, vf_coef, max_grad_norm, gpu=0):
        sess = tf.get_default_session()

        with tf.device('/device:GPU:'+ str(gpu)):
            if isinstance(ac_space, spaces.Box):
                nact = np.sum(ac_space.shape) *2
            else:
                nact = np.sum(ac_space.nvec)

            X = tf.placeholder(tf.float32, (nbatch_act, ob_space.shape[0]), "X")
            act_model = policy( X, sess, nact, ac_space, nbatch_act, 1, nlstm=512, reuse=False) # (sess, ob_space, ac_space, [nbatch_act], 1, reuse=False)
            X = tf.placeholder(tf.float32, (nbatch_train, ob_space.shape[0]), "X_1")
            train_model = policy( X, sess, nact, ac_space, nbatch_train, nsteps,nlstm=512, reuse=True)#(sess, ob_space, ac_space, [nbatch_train], nsteps, reuse=True)

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
            advs = returns - values
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            td_map = {train_model.X:obs, A:actions, ADV:advs, R:returns, LR:lr,
                    CLIPRANGE:cliprange, OLDNEGLOGPAC:neglogpacs, OLDVPRED:values, ENT_COEFF:ent_coeff}
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

        self.train = train
        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value
        self.initial_state = act_model.initial_state
        # self.save = save
        # self.load = load
        tf.global_variables_initializer().run(session=sess) #pylint: disable=E1101

class Runner(object):

    def __init__(self, *, env, model, model2, nsteps, gamma, lam):
        self.env = env
        self.model = model
        self.model2 = model2

        nenv = env.num_envs

        self.obs = np.zeros((nenv,) + env.observation_space.shape, dtype=model.train_model.X.dtype.name)
        self.obs[:] = env.reset()
        self.gamma = gamma
        self.lam = lam
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]

    def run(self, lrnow):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [], [], [], [], [], []
        mb_states = self.states
        epinfos = []
        genEnvLosses = []
        imgs = []

        for _ in range(self.nsteps):
            actions, values, self.states, neglogpacs = self.model.step(self.obs, self.states, self.dones)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            # ob_img = self.model2.step(self.obs[:]) # TODO was this necassary?
            imgs = []
            test_obs = []  # TODO remove test stuff
            test_imgs = []
            for info in infos:
                imgs.append(info.get("img"))
                test_obs.append(info.get("test_ob"))
                test_imgs.append(info.get("test_img"))
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            # loss = self.model2.train(self.obs[:], imgs, lrnow)  # TODO anneal lr
            loss = self.model2.train(test_obs, test_imgs, lrnow)
            genEnvLosses.append(loss)
            mb_rewards.append(rewards)
        #batch of steps to batch of rollouts
        mb_genEnvLosses = np.asarray(genEnvLosses, dtype=np.float32)
        # mb_imgs = np.asarray(imgs, dtype=np.uint8)
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, self.states, self.dones)
        #discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values

        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
            mb_states, epinfos,  mb_genEnvLosses)

# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

def constfn(val):
    def f(_):
        return val
    return f

def learn(*, policy, env, nsteps, total_timesteps, ent_coef, lr,
            vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,
            log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
            save_interval=0, loadModel=None, gpu=0):

    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(ent_coef, float): ent_coef = constfn(ent_coef)
    else: assert callable(ent_coef)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)
    total_timesteps = int(total_timesteps)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches

    make_model = lambda : Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                    nsteps=nsteps, vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm, gpu=gpu)

    import gym
    from Redbird_AI.common.policies import RevConv
    from Redbird_AI.modelEnv import Model2
    import math
    make_model2 = lambda : Model2(policy=RevConv,
                    ob_space=gym.spaces.Box(np.array([0, 0, 0, False]), np.array([20.0, 20.0, math.pi*2, True]), dtype=np.float32),
                    ac_space=gym.spaces.Box(0, 255, [64, 64]), nbatch_act=nenvs, nbatch_train=nbatch_train,
                    nsteps=10, #TODO remove hardcode
                    max_grad_norm=max_grad_norm)

    model = make_model()
    model2 = make_model2()
    writer = tf.summary.FileWriter(logger.get_dir() + '/tb/', tf.get_default_graph())
    writer.close()
    if loadModel is not None:
        env.ob_rms, env.ret_rms = load_model(loadModel)

    runner = Runner(env=env, model=model, model2=model2, nsteps=nsteps, gamma=gamma, lam=lam)

    epinfobuf = deque(maxlen=100)
    tfirststart = time.time()

    nupdates = total_timesteps//nbatch
    for update in range(1, nupdates+1):
        assert nbatch % nminibatches == 0
        nbatch_train = nbatch // nminibatches
        tstart = time.time()
        frac = 1.0 - (update - 1.0) / nupdates
        lrnow = lr(frac)
        ent_coeff_now = ent_coef(frac)
        cliprangenow = cliprange(frac)
        obs, returns, masks, actions, values, neglogpacs, states_agent, epinfos, lossvals_genEnv = runner.run(lrnow) #pylint: disable=E0632
        epinfobuf.extend(epinfos)
        mblossvals_agent = []
        mblossvals_genEnv = []
        states_genEnv = None #TODO remove this and make recurrent support genEnv
        if states_agent is None: # nonrecurrent version
            inds = np.arange(nbatch)
            for _ in range(noptepochs):
                np.random.shuffle(inds)
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mblossvals_agent.append(model.train(lrnow, cliprangenow, *slices, ent_coeff_now))
        else: # recurrent version
            assert nenvs % nminibatches == 0
            envsperbatch = nenvs // nminibatches
            envinds = np.arange(nenvs)
            flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
            envsperbatch = nbatch_train // nsteps
            for _ in range(noptepochs):
                np.random.shuffle(envinds)
                for start in range(0, nenvs, envsperbatch):
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    mbflatinds = flatinds[mbenvinds].ravel()
                    slices = (arr[mbflatinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mbstates = states_agent[mbenvinds]
                    mblossvals_agent.append(model.train(lrnow, cliprangenow, *slices, ent_coeff_now, mbstates))
        # if states_genEnv is None: # nonrecurrent version
        #     inds = np.arange(nbatch)
        #     for _ in range(noptepochs):
        #         np.random.shuffle(inds)
        #         for start in range(0, nbatch, nbatch_train):
        #             end = start + nbatch_train
        #             mbinds = inds[start:end]
        #             slices = (arr[mbinds] for arr in (obs, tru_imgs))
        #             mblossvals_genEnv.append(model2.train(*slices, lrnow))
        # else: # recurrent version  #TODO make recurrent support genEnv
        #     assert nenvs % nminibatches == 0
        #     envsperbatch = nenvs // nminibatches
        #     envinds = np.arange(nenvs)
        #     flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
        #     envsperbatch = nbatch_train // nsteps
        #     for _ in range(noptepochs):
        #         np.random.shuffle(envinds)
        #         for start in range(0, nenvs, envsperbatch):
        #             end = start + envsperbatch
        #             mbenvinds = envinds[start:end]
        #             mbflatinds = flatinds[mbenvinds].ravel()
        #             slices = (arr[mbflatinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
        #             mbstates = states_agent[mbenvinds]
        #             mblossvals_agent.append(model.train(lrnow, cliprangenow, *slices, ent_coeff_now, mbstates))
        lossvals_agent = np.mean(mblossvals_agent, axis=0)
        # lossvals_genEnv = np.mean(mblossvals_genEnv, axis=0)
        tnow = time.time()
        fps = int(nbatch / (tnow - tstart))
        if update % log_interval == 0 or update == 1:
            ev = explained_variance(values, returns)
            logger.logkv("serial_timesteps", update*nsteps)
            logger.logkv("nupdates", update)
            logger.logkv("total_timesteps", update*nbatch)
            logger.logkv("fps", fps)
            logger.logkv("explained_variance", float(ev))
            logger.logkv("lr", float(lrnow))
            logger.logkv("ent_coef", float(ent_coeff_now))
            logger.logkv("genEnvLosses", safemean(lossvals_genEnv))
            try:
                for key in epinfobuf[0].keys():
                    logger.logkv(key, safemean([epinfo[key] for epinfo in epinfobuf]))
            except:
                print("bummer") #TODO: cleanup
            logger.logkv('time_elapsed', tnow - tfirststart)
            for (lossval, lossname) in zip(lossvals_agent, model.loss_names):
                logger.logkv(lossname, lossval)
            logger.dumpkvs()
        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir():
            save_model(update, env.ob_rms, env.ret_rms)
    env.close()

def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)
