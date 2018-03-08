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
from collections import Counter

class Model(object):
    def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                nsteps, ent_coef, vf_coef, max_grad_norm, gpu):
        sess = tf.get_default_session()

        with tf.device('/device:GPU:'+ str(gpu)):
            ob_shape = (nbatch_act, ob_space.shape[0])
            nact = np.sum(ac_space.nvec)
            X = tf.placeholder(tf.float32, (nbatch_act, ob_space.shape[0]), "X")

            act_model = policy( X, sess, nact, ac_space, reuse=False) # (sess, ob_space, ac_space, [nbatch_act], 1, reuse=False)
            X = tf.placeholder(tf.float32, (nbatch_train, ob_space.shape[0]), "X_1")
            train_model = policy( X, sess, nact, ac_space, reuse=True)#(sess, ob_space, ac_space, [nbatch_train], nsteps, reuse=True)

            A = train_model.pdtype.sample_placeholder([None])
            ADV = tf.placeholder(tf.float32, [None])
            R = tf.placeholder(tf.float32, [None])
            OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
            OLDVPRED = tf.placeholder(tf.float32, [None])
            LR = tf.placeholder(tf.float32, [])
            CLIPRANGE = tf.placeholder(tf.float32, [])

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
            pi_loss = pg_loss - entropy * ent_coef
            vf_loss = vf_loss_ * vf_coef
            general_loss = pi_loss + vf_loss

            # general_params = tf.trainable_variables("general_layers")
            # pi_params = tf.trainable_variables("pi_layers")
            # vf_params = tf.trainable_variables("vf_layers")
            params = tf.trainable_variables("model")

            # general_grads = tf.gradients(general_loss, general_params)
            # pi_grads = tf.gradients(pi_loss, pi_params)
            # vf_grads = tf.gradients(vf_loss, vf_params)
            grads = tf.gradients(general_loss, params)

            if max_grad_norm is not None:
                general_grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
            grads = list(zip(general_grads, params))

            # if max_grad_norm is not None:
            #     pi_grads, _grad_norm = tf.clip_by_global_norm(pi_grads, max_grad_norm)
            # pi_grads = list(zip(pi_grads, pi_params))
            #
            # if max_grad_norm is not None:
            #     vf_grads, _grad_norm = tf.clip_by_global_norm(vf_grads, max_grad_norm)
            # vf_grads = list(zip(vf_grads, vf_params))

            trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5, name="adam")
            # _train = trainer.apply_gradients(general_grads + vf_grads + pi_grads)
            _train = trainer.apply_gradients(grads)

        def train(lr, cliprange, obs, returns, masks, actions, values, neglogpacs, states=None):
            advs = returns - values
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            td_map = {train_model.X:obs, A:actions, ADV:advs, R:returns, LR:lr,
                    CLIPRANGE:cliprange, OLDNEGLOGPAC:neglogpacs, OLDVPRED:values}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            stuff =  sess.run(
                [pg_loss, vf_loss, entropy, approxkl, clipfrac, general_loss, _train],#, summaries],
                td_map
            )[:-1]
            # logger.Logger.CURRENT.writer.add_summary(stuff[-1], global_step=logger.Logger.CURRENT.step)
            return stuff #[:-1]
        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac', 'total_loss']

        # def save(save_path):
        #
        #     # os.makedirs(os.path.dirname(self.this_test + '/model/model.ckpt'), exist_ok=True)
        #     os.makedirs(os.path.dirname(save_path), exist_ok=True)
        #     saver = tf.train.Saver(var_list=tf.global_variables())
        #     saver.save(tf.get_default_session(), save_path)
        #     # ps = sess.run(general_params + vf_params + pi_params)
        #     # joblib.dump(ps, save_path)
        #
        #
        # def load(load_path):
        #     print('loading old model')
        #     # from tensorflow.contrib.framework.python.framework.checkpoint_utils import  list_variables
        #     var_list = tf.global_variables() # trainable_variables()
        #     for vars in var_list:
        #         try:
        #             saver = tf.train.Saver({vars.name[:-2]: vars})  # the [:-2] is kinda jerry-rigged but ..
        #             saver.restore(tf.get_default_session(), load_path + '.ckpt')
        #             print("found " + vars.name)
        #         except:
        #             print("couldn't find " + vars.name)
        #     print('finished loading model')
            # saver = tf.train.Saver()
            # try:
            #     saver.restore(tf.get_default_session(), load_path)
            # except tf.errors.InvalidArgumentError:
            #     print('couldn''t find a valid model at that location')

            # loaded_params = joblib.load(load_path)
            # restores = []
            # for p, loaded_p in zip(general_params + vf_params + pi_params, loaded_params):
            #     restores.append(p.assign(loaded_p))
            # sess.run(restores)
            # # If you want to load weights, also save/load observation scaling inside VecNormalize

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

    def __init__(self, *, env, model, nsteps, gamma, lam):
        self.env = env
        self.model = model

        nenv = env.num_envs

        self.obs = np.zeros((nenv,) + env.observation_space.shape, dtype=model.train_model.X.dtype.name)
        self.obs[:] = env.reset()
        self.gamma = gamma
        self.lam = lam
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        mb_states = self.states
        epinfos = []
        #rews = []
        for _ in range(self.nsteps):
            actions, values, self.states, neglogpacs = self.model.step(self.obs, self.states, self.dones)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)

            for info in infos:

                maybeepinfo = info.get('episode')
                if maybeepinfo:
                    epinfos.append(maybeepinfo)
               # mayberewsinfo = info.get('rews')
               # if mayberewsinfo : rews.append(mayberewsinfo)

            #rews.append(sum(list(Counter(info.get('rews')) for info in infos)))
            mb_rewards.append(rewards)
        #batch of steps to batch of rollouts
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
            mb_states, epinfos)

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
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)
    total_timesteps = int(total_timesteps)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches

    make_model = lambda : Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                    nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm, gpu=gpu)
    # if save_interval and logger.get_dir():
    #     import cloudpickle
    #     with open(osp.join(logger.get_dir(), 'make_model.pkl'), 'wb') as fh:
    #         fh.write(cloudpickle.dumps(make_model))
    model = make_model()
    writer = tf.summary.FileWriter(logger.get_dir() + '/tb/', tf.get_default_graph())
    writer.close()
    if loadModel is not None:
        env.ob_rms, env.ret_rms = load_model(loadModel)
        # model.load(loadModel)
        # import pickle
        # try:
        #     with open(loadModel + '.pik', 'rb') as f:
        #         env.ob_rms, env.ret_rms = pickle.load(f)
        #     print('found observation scaling')
        # except:
        #     print('could not find observation scaling')
        # # data =  m4p.loadmat(osp.join(logger.get_dir(), 'checkpoints/obs_scaling.mat'))
    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)

    epinfobuf = deque(maxlen=100)
    tfirststart = time.time()

    nupdates = total_timesteps//nbatch
    for update in range(1, nupdates+1):
        assert nbatch % nminibatches == 0
        nbatch_train = nbatch // nminibatches
        tstart = time.time()
        frac = 1.0 - (update - 1.0) / nupdates
        lrnow = lr(frac)
        cliprangenow = cliprange(frac)
        obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run() #pylint: disable=E0632
        epinfobuf.extend(epinfos)
        mblossvals = []
        if states is None: # nonrecurrent version
            inds = np.arange(nbatch)
            for _ in range(noptepochs):
                np.random.shuffle(inds)
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices))
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
                    mbstates = states[mbenvinds]
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices, mbstates))

        lossvals = np.mean(mblossvals, axis=0)
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
            logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
            logger.logkv('game_reward', safemean([epinfo['game'] for epinfo in epinfobuf]))
            logger.logkv('direction_reward', safemean([epinfo['direction'] for epinfo in epinfobuf]))
            logger.logkv('selection_reward', safemean([epinfo['selection'] for epinfo in epinfobuf]))
            logger.logkv('end_reward', safemean([epinfo['end'] for epinfo in epinfobuf]))
            logger.logkv('time_elapsed', tnow - tfirststart)
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                logger.logkv(lossname, lossval)
            logger.dumpkvs()
        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir():
            save_model(update, env.ob_rms, env.ret_rms)
            # checkdir = osp.join(logger.get_dir(), 'checkpoints')
            # os.makedirs(checkdir, exist_ok=True)
            # savepath = osp.join(checkdir, '%.5i.ckpt'%update)
            # print('Saving to', savepath)
            # model.save(savepath)
            # import pickle
            # data = env.ob_rms
            # with open(osp.join(checkdir, '%.5i.pik'%update), 'wb') as f:
            #     pickle.dump([env.ob_rms, env.ret_rms], f, -1)
    env.close()

def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)
