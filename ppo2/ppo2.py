import os
import time
import joblib
import numpy as np
import os.path as osp
import tensorflow as tf
from baselines import logger
from collections import deque
from baselines.common import explained_variance
from baselines.common.mpi_adam import MpiAdam
from baselines.common import tf_util as U
from baselines.common.mpi_moments import mpi_moments
from tensorflow.python.client import timeline

class Model(object):
    def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                nsteps, ent_coef, vf_coef, max_grad_norm):
        sess = tf.get_default_session()

        act_model = policy(sess, ob_space, ac_space, reuse=False)
        train_model = policy(sess, ob_space, ac_space, reuse=True) # was true -ben

        A = train_model.pdtype.sample_placeholder([None])
        ADV = tf.placeholder(tf.float32, [None], name="ADV")
        R = tf.placeholder(tf.float32, [None], name="REW")
        OLDNEGLOGPAC = tf.placeholder(tf.float32, [None], name="OLDNEGLOGPAC")
        OLDVPRED = tf.placeholder(tf.float32, [None],name = "OLDVPRED")
        LR = tf.placeholder(tf.float32, [])
        CLIPRANGE = tf.placeholder(tf.float32, [])

        neglogpac = train_model.pd.neglogp(A)
        entropy = tf.reduce_mean(train_model.pd.entropy())

        vpred = train_model.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        vf_losses1 = tf.square(vpred - R)
        vf_losses2 = tf.square(vpredclipped - R)
        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
        pg_losses = -ADV * ratio
        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef
        with tf.variable_scope('model'):
            params = tf.trainable_variables()
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))

        # from  ppo1
        adam = MpiAdam(params, epsilon=1e-5)
        U.initialize()
        adam.sync()
        # _train = adam.update(g, optim_stepsize * cur_lrmult)
        # trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        # _train = trainer.apply_gradients(grads)

        def train(lr, cliprange, obs, returns, masks, actions, values, neglogpacs, states=None):
            advs = returns - values
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            td_map = {train_model.X:obs, A:actions, ADV:advs, R:returns, LR:lr,
                    CLIPRANGE:cliprange, OLDNEGLOGPAC:neglogpacs, OLDVPRED:values}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            *losses, g = sess.run(
                # [pg_loss, vf_loss, entropy, approxkl, clipfrac, _train],
                [pg_loss, vf_loss, entropy, approxkl, clipfrac, U.flatgrad(loss, params)],
                td_map, options=run_options, run_metadata=run_metadata
            ) # [:-1]
            adam.update(g, 2.5e-4 * 0.95) #TODO: justify the use of these values
            # Create the Timeline object, and write it to a json
            tl = timeline.Timeline(run_metadata.step_stats)
            ctf = tl.generate_chrome_trace_format()
            # os.makedirs('/tmp/timeline/')
            # with open('/tmp/timeline/timeline.json', 'w') as f:
            #     f.write(ctf)
            return losses
        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']

        def save(save_path):
            ps = sess.run(params)
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            sess.run(restores)
            # If you want to load weights, also save/load observation scaling inside VecNormalize
        self.sync = adam.sync
        self.train = train
        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value
        self.initial_state = act_model.initial_state
        self.save = save
        self.load = load
        tf.global_variables_initializer().run(session=sess) #pylint: disable=E1101

class Runner(object):

    def __init__(self, *, env, model, nsteps, gamma, lam, writer, earlyTermT_ms, rank, render=False):
        self.env = env
        self.model = model
        self.obs = env.reset()
        self.gamma = gamma
        self.lam = lam
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = False
        self.ep_rew = 0
        self.ep_num = 0
        self.earlyTermT_ms = earlyTermT_ms
        self.writer = writer
        self.render = render
        self.rank = rank

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        mb_states = self.states
        epinfos = []
        for _ in range(self.nsteps):
            ac, vpred, self.states, neglogpacs = self.model.step(self.obs, self.states, self.dones)
            mb_obs.append(self.obs.copy())
            mb_actions.append(ac)
            mb_values.append(vpred)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)
            self.obs, reward, self.dones, info = self.env.step(ac)

            if self.render and self.rank == 0:
                self.env.render()

            if self.earlyTermT_ms is not None and info["time_ms"] >= self.earlyTermT_ms:
                self.dones = True
                self.obs = self.env.reset()
                self.states = self.model.initial_state
            # for info in infos:
            #     maybeepinfo = info.get('episode')
            #     if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(reward)
            self.ep_rew += reward
            if self.dones:
                print(self.ep_rew)
                summary = tf.Summary(value=[tf.Summary.Value(tag="rew", simple_value=self.ep_rew)])
                self.writer.add_summary(summary, self.ep_num)
                self.writer.flush()
                self.ep_num += 1
                self.ep_rew = 0
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
        mb_advs = [] # ben # np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            # mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
            lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
            mb_advs.append(lastgaelam)
        mb_returns = np.array(mb_advs) + mb_values


        # return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
        #     mb_states, epinfos)
        return mb_obs, mb_returns[:,0], mb_dones, mb_actions, mb_values[:,0], mb_neglogpacs[:,0]
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

class Redbird_Pposgd2():

    def __init__(self, rank, this_test, last_test, render, earlyTermT_ms=None):
        self.rank = rank
        self.earlyTermT_ms = earlyTermT_ms
        self.this_test = this_test
        self.last_test = last_test
        sess = tf.get_default_session()
        self.writer = tf.summary.FileWriter(self.this_test + '/rank_' + str(self.rank), sess.graph)
        self.render = render

    def learn(self, *, policy, env, nsteps, total_timesteps, ent_coef, lr,
                vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,
                log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
                save_interval=0):

        if isinstance(lr, float): lr = constfn(lr)
        else: assert callable(lr)
        if isinstance(cliprange, float): cliprange = constfn(cliprange)
        else: assert callable(cliprange)
        total_timesteps = int(total_timesteps)

        nenvs = 1 # ben
        ob_space = env.observation_space
        ac_space = env.action_space
        nbatch = nenvs * nsteps
        nbatch_train = nbatch // nminibatches
        nbatch_train = nsteps // nminibatches

        make_model = lambda : Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                        nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                        max_grad_norm=max_grad_norm)
        # if save_interval and logger.get_dir():
        #     import cloudpickle
        #     with open(osp.join(logger.get_dir(), 'make_model.pkl'), 'wb') as fh:
        #         fh.write(cloudpickle.dumps(make_model))
        model = make_model()
        runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam, writer=self.writer,
                        earlyTermT_ms=self.earlyTermT_ms, rank=self.rank, render=self.render)

        model.sync()

        epinfobuf = deque(maxlen=100)
        tfirststart = time.time()

        nupdates = total_timesteps//nbatch

        # with tf.contrib.tfprof.ProfileContext('/tmp/train_dir') as pctx:
        for update in range(1, nupdates+1):
            assert nbatch % nminibatches == 0
            nbatch_train = nbatch // nminibatches
            tstart = time.time()
            frac = 1.0 - (update - 1.0) / nupdates
            lrnow = lr(frac)
            cliprangenow = cliprange(frac)
            # obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run() #pylint: disable=E0632
            obs, returns, masks, actions, values, neglogpacs= runner.run()  # pylint: disable=E0632
            states = None # ben because non-recurrent right now
            # epinfobuf.extend(epinfos)
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
            meanlosses, _, _ = mpi_moments(lossvals, axis=0)
            tnow = time.time()
            fps = int(nbatch / (tnow - tstart))
            if update % log_interval == 0 or update == 1:
                ev = explained_variance(values, returns)
                # logger.logkv("serial_timesteps", update*nsteps)
                # logger.logkv("nupdates", update)
                # logger.logkv("total_timesteps", update*nbatch)
                # logger.logkv("fps", fps)
                # logger.logkv("explained_variance", float(ev))
                # logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
                # logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
                # logger.logkv('time_elapsed', tnow - tfirststart)
                summary = tf.Summary(value=[tf.Summary.Value(tag="fps", simple_value=fps)])
                self.writer.add_summary(summary, update)
                summary = tf.Summary(value=[tf.Summary.Value(tag="explained_variance", simple_value=float(ev))])
                self.writer.add_summary(summary, update)

                for (lossval, lossname) in zip(lossvals, model.loss_names):
                    # logger.logkv(lossname, lossval)
                    summary = tf.Summary(value=[tf.Summary.Value(tag=lossname, simple_value=lossval)])
                    self.writer.add_summary(summary, update*nsteps)
                # logger.dumpkvs()
            # if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir():
            #     checkdir = osp.join(logger.get_dir(), 'checkpoints')
            #     os.makedirs(checkdir, exist_ok=True)
            #     savepath = osp.join(checkdir, '%.5i'%update)
            #     print('Saving to', savepath)
            #     model.save(savepath)
        env.close()

def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)
