import tensorflow as tf
import numpy as np
from baselines.common import tf_util as U
from baselines.common import zipsame, Dataset
from baselines.common.mpi_adam import MpiAdam
import time
from mpi4py import MPI
from baselines.common.mpi_moments import mpi_moments
import os
from baselines.common import explained_variance
from baselines import logger
import os.path as osp
import joblib


REWARD_SCALE = 10 # get rid of later

class RedbirdPposgd():
    def __init__(self, rank, this_test, last_test, earlyTermT_ms=None):
        self.rank = rank
        self.earlyTermT_ms = earlyTermT_ms
        self.this_test = this_test
        self.last_test = last_test
        sess = tf.get_default_session()
        self.writer = tf.summary.FileWriter(self.this_test + '/rank_' + str(self.rank), sess.graph)

    def traj_segment_generator(self, pi, env, horizon, stochastic, render=False):
        t = 0
        ac = env.action_space.sample() # not used, just so we have the datatype
        new = True # marks if we're on first timestep of an episode
        ob = env.reset()

        cur_ep_ret = 0 # return in current episode
        cur_ep_len = 0 # len of current episode
        ep_num = 0
        ep_rets = [] # returns of completed episodes in this segment
        ep_lens = [] # lengths of ...

        # Initialize history arrays
        obs = np.array([ob for _ in range(horizon)])
        rews = np.zeros(horizon, 'float32')
        prev_vpreds = np.zeros(horizon, 'float32')
        vpreds = np.zeros(horizon, 'float32')
        news = np.zeros(horizon, 'int32')
        acs = np.array([ac for _ in range(horizon)])
        prevacs = acs.copy()
        self.states = pi.initial_state
        self.done = False
        while True:
            prevac = ac
            # ac, vpred = pi.act(stochastic, ob)
            ac, vpred, self.states, _ = pi.step(ob, self.states, self.done)

            #tensorboard logging

            if t > 0 and t % horizon == 0:
                yield {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
                        "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                        "ep_rets" : ep_rets, "ep_lens" : ep_lens}
                # Be careful!!! if you change the downstream algorithm to aggregate
                # several of these batches, then be sure to do a deepcopy
                ep_rets = []
                ep_lens = []
            i = t % horizon
            obs[i] = ob
            vpreds[i] = vpred
            news[i] = new
            acs[i] = ac
            prevacs[i] = prevac

            ac = np.expand_dims(ac, 0)
            ob, rew, new, info = env.step(ac)
            # ob = ob [0,:] #ben
            # if self.earlyTermT_ms is not None and info["time_ms"] >= self.earlyTermT_ms:
            #     new = True

            # rew = rew * REWARD_SCALE  # ben
            rews[i] = rew
            cur_ep_ret += rew
            cur_ep_len += 1

            if render and self.rank == 0:
                env.render()

            #more tensorboard logging stuff
            #here

            if new:
                ep_num += 1
                summary = tf.Summary(value=[tf.Summary.Value(tag="rew", simple_value=cur_ep_ret)])
                self.writer.add_summary(summary, ep_num)
                # self.writer.flush() #ben

                ep_rets.append(cur_ep_ret)
                ep_lens.append(cur_ep_len)
                cur_ep_ret = 0
                cur_ep_len = 0
                ob = env.reset()
            t += 1
    def add_vtarg_and_adv(self, seg, gamma, lam):
        """
        Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
        """
        new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
        vpred = np.append(seg["vpred"], seg["nextvpred"])

        T = len(seg["rew"])
        seg["adv"] = gaelam = np.empty(T, 'float32')
        rew = seg["rew"]
        lastgaelam = 0
        for t in reversed(range(T)):
            nonterminal = 1-new[t+1]
            delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
            gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        seg["tdlamret"] = seg["adv"] + seg["vpred"]

    def constfn(self, val):
        def f(_):
            return val

        return f

    def learn(self, env, policy_func, *,
            timesteps_per_actorbatch, # timesteps per actor per update
            clip_param, entcoeff, vf_coef, # clipping parameter epsilon, entropy coeff
            optim_epochs, optim_stepsize, optim_batchsize,# optimization hypers
            gamma, lam, # advantage estimation
            max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
            callback=None, # you can do anything in the callback, since it takes locals(), globals()
            adam_epsilon=1e-5,
            schedule='constant', # annealing for stepsize parameters (epsilon and adam),
            render, loadModel, lr, cliprange=0.2, save_interval=200
            ):
        if isinstance(lr, float): lr = self.constfn(lr)
        else: assert callable(lr)
        if isinstance(cliprange, float): cliprange = self.constfn(cliprange)
        else: assert callable(cliprange)

        ob_space = env.observation_space
        ac_space = env.action_space
        pi = policy_func("pi", ob_space, ac_space, False)  # Construct network for new policy
        oldpi = policy_func("oldpi", ob_space, ac_space, True)  # Network for old policy
        atarg = tf.placeholder(dtype=tf.float32, shape=[None],
                               name="atarg")  # Target advantage function (if applicable)
        ret = tf.placeholder(dtype=tf.float32, shape=[None], name="ret")  # Empirical return TODO: what is this? -ben

        lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule
        clip_param = clip_param * lrmult # Annealed cliping parameter epislon

        ob = U.get_placeholder_cached(name="X")
        ac = pi.pdtype.sample_placeholder([None])

        OLDVPRED = tf.placeholder(tf.float32, [None],name = "OLDVPRED") # from ppo2
        CLIPRANGE = tf.placeholder(tf.float32, [])

        kloldnew = oldpi.pd.kl(pi.pd)
        meankl = tf.reduce_mean(kloldnew)
        ent = pi.pd.entropy()
        meanent = tf.reduce_mean(ent)
        pol_entpen = (-entcoeff)*meanent

        ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) #pnew / pold
        surr1 = ratio * atarg # surrogate from conservative policy iteration
        surr2 = tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE) * atarg
        pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2))  # PPO's pessimistic surrogate (L^CLIP)
        # vf_loss = tf.reduce_mean(tf.square(pi.vpred - ret))
        # vpredclipped = OLDVPRED + tf.clip_by_value(pi.vpred - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        vpredclipped = OLDVPRED + tf.clip_by_value(pi.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        # vf_losses1 = tf.square(pi.vpred- ret)
        vf_losses1 = tf.square(pi.vf - ret)
        vf_losses2 = tf.square(vpredclipped - ret)
        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
        total_loss = pol_surr + pol_entpen + vf_loss * vf_coef
        losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent, total_loss]
        loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent", "total"]

        # var_list = pi.get_trainable_variables()
        var_list = tf.trainable_variables(scope="model")
        lossandgrad = U.function([ob, ac, atarg, ret, lrmult, OLDVPRED, CLIPRANGE], losses + [U.flatgrad(total_loss, var_list)])
        adam = MpiAdam(var_list, epsilon=adam_epsilon)

        # assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        #     # for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
        #     for (oldv, newv) in zipsame()
        compute_losses = U.function([ob, ac, atarg, ret, lrmult, OLDVPRED, CLIPRANGE], losses)

        U.initialize()
        adam.sync()
        # if save_interval and logger.get_dir():
        #     import cloudpickle
        #     with open(osp.join(logger.get_dir(), 'make_model.pkl'), 'wb') as fh:
        #         fh.write(cloudpickle.dumps(make_model))
        if loadModel is not None:
            print('loading old model')
            var_list = tf.trainable_variables()
            for vars in var_list:
                try:
                    saver = tf.train.Saver({vars.name[:-2]: vars})  # the [:-2] is kinda jerry-rigged but ..
                    saver.restore(tf.get_default_session(), loadModel + '.ckpt')
                    print("found " + vars.name)
                except:
                    print("couldn't find " + vars.name)
            print('finished loading model')
            # If you want to load weights, also save/load observation scaling inside VecNormalize

        # if not newModel:
        #     saver = tf.train.Saver()
        #     try:
        #         saver.restore(tf.get_default_session(), self.last_test + '/model/model.ckpt')
        #     except tf.errors.InvalidArgumentError:
        #         print('couldn''t find a valid model at that location')

        seg_gen = self.traj_segment_generator(pi, env, timesteps_per_actorbatch, stochastic=True, render=render)

        episodes_so_far = 0
        timesteps_so_far = 0
        iters_so_far = 0
        tstart = time.time()

        assert sum([max_iters>0, max_timesteps>0, max_episodes>0, max_seconds>0])==1, "Only one time constraint permitted"

        while True:
            if callback: callback(locals(), globals())
            if max_timesteps and timesteps_so_far >= max_timesteps:
                break
            elif max_episodes and episodes_so_far >= max_episodes:
                break
            elif max_iters and iters_so_far >= max_iters:
                break
            elif max_seconds and time.time() - tstart >= max_seconds:
                break

            # if schedule == 'constant':
            #     cur_lrmult = 1.0
            # elif schedule == 'linear':
            #     cur_lrmult =  max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
            # else:
            #     raise NotImplementedError

            if self.rank == 0:
                print("********** Iteration %i ************"%iters_so_far)
            seg = seg_gen.__next__()

            self.add_vtarg_and_adv(seg, gamma, lam)
            seg["ob"] = seg["ob"][:,0,:] #ben
            ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
            vpredbefore = seg["vpred"]  # predicted value function before udpate
            atarg = (atarg - atarg.mean()) / atarg.std()  # standardized advantage function estimate
            d = Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret, vpredbefore=vpredbefore), shuffle=True)
            optim_batchsize = optim_batchsize or ob.shape[0]
            frac = 1.0 - (timesteps_so_far - 1.0) / max_timesteps
            lrnow = lr(frac)
            cliprangenow = cliprange(frac)

            tstart = time.time()

            if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob)  # update running mean/std for policy

            # assign_old_eq_new()  # set old parameter values to new parameter values

            # Here we do a bunch of optimization epochs over the data
            for _ in range(optim_epochs):
                losses = [] # list of tuples, each of which gives the loss for a minibatch
                for batch in d.iterate_once(optim_batchsize):
                    #    lossandgrad = U.function([ob, ac, atarg, ret, lrmult], losses + [U.flatgrad(total_loss, var_list)])
                    *newlosses, g = lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], lrnow, batch["vpredbefore"], cliprangenow)
                    adam.update(g, optim_stepsize * lrnow)
                    losses.append(newlosses)
            # if iters_so_far % 25 == 0:
            #     os.makedirs(os.path.dirname(self.this_test + '/model/model.ckpt'), exist_ok=True)
            #     saver = tf.train.Saver()
            #     saver.save(tf.get_default_session(), self.this_test + '/model/model.ckpt')
            if save_interval and (iters_so_far % save_interval == 0 or iters_so_far == 1) and logger.get_dir():
                checkdir = osp.join(logger.get_dir(), 'checkpoints')
                os.makedirs(checkdir, exist_ok=True)
                savepath = osp.join(checkdir, '%.5i.ckpt' % iters_so_far)
                print('Saving to', savepath)
                os.makedirs(os.path.dirname(savepath), exist_ok=True)
                saver = tf.train.Saver(var_list=tf.trainable_variables())
                var_list = tf.trainable_variables()
                saver.save(tf.get_default_session(), savepath)
                import pickle
                data = env.ob_rms
                with open(osp.join(checkdir, '%.5i.pik' % iters_so_far), 'wb') as f:
                    pickle.dump([env.ob_rms, env.ret_rms], f, -1)


            losses = []
            for batch in d.iterate_once(optim_batchsize):
                newlosses = compute_losses(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], lrnow, batch["vpredbefore"], cliprangenow)
                losses.append(newlosses)
                # if losses[-1] < self.min_loss:
                #     U.save_state('/tmp/models/' + MODEL_NAME + '.ckpt')
                #     self.min_loss = losses[-1]
            meanlosses,_,_ = mpi_moments(losses, axis=0)
            tnow = time.time()
            fps = float(1 / (tnow - tstart))
            if self.rank == 0:
                ev = explained_variance(seg["vpred"], seg["tdlamret"])
                summary = tf.Summary(value=[tf.Summary.Value(tag="iters_per_sec", simple_value=fps)])
                self.writer.add_summary(summary,  iters_so_far)
                summary = tf.Summary(value=[tf.Summary.Value(tag="explained_variance", simple_value=ev)])
                self.writer.add_summary(summary, iters_so_far)
                for (lossval, name) in zipsame(meanlosses, loss_names):
                    summary = tf.Summary(value=[tf.Summary.Value(tag="loss_"+name, simple_value=lossval)])
                    self.writer.add_summary(summary, iters_so_far)
                    summary = tf.Summary(value=[tf.Summary.Value(tag="lr", simple_value=lrnow)])
                    self.writer.add_summary(summary, iters_so_far)

            lrlocal = (seg["ep_lens"], seg["ep_rets"]) # local values
            listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
            lens, rews = map(self.flatten_lists, zip(*listoflrpairs))
            episodes_so_far += len(lens)
            timesteps_so_far += sum(lens)
            iters_so_far += 1

    def flatten_lists(self, listoflists):
        return [el for list_ in listoflists for el in list_]