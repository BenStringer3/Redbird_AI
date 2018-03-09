#!/usr/bin/env python3
import sys
sys.path.append('/home/redbird_general/Desktop/Redbird_AI2/')

from baselines import logger
from Redbird_AI.ppo2.redbird_ppo2 import learn
from Redbird_AI.common.policies import  MlpPolicy3, MlpPolicy4, MlpPolicy5
import multiprocessing
import tensorflow as tf
import gym
from baselines.common import set_global_seeds
from Redbird_AI.common.rb_monitor import RB_Monitor
from baselines.bench import Monitor
from Redbird_AI.common.cmd_util import iarc_arg_parser, make_env
import os


def make_IARC_env(env_id, num_env, seed, earlyTerminationTime_ms, wrapper_kwargs=None, start_index=0):
    import gym
    from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

    """
    Create a wrapped, monitored SubprocVecEnv for Atari.
    """
    if wrapper_kwargs is None: wrapper_kwargs = {}
    def make_env(rank): # pylint: disable=C0111
        def _thunk():
            env = gym.make(env_id)
            env.seed(seed + 1000*rank)
            env.env.earlyTerminationTime_ms = earlyTerminationTime_ms
            env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
            env = RB_Monitor(env)
            return env
        return _thunk
    set_global_seeds(seed)
    envs = [make_env(i + start_index) for i in range(num_env)]
    ret= SubprocVecEnv(envs)
    return ret

def train(env_id, num_timesteps, seed, policy, earlyTerminationTime_ms, loadModel, nenv, ent_coef, initial_lr=2.5e-4, gpu=0, anneal_ent_coef=0):
    from baselines.common.vec_env.vec_normalize import VecNormalize
    from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    config.gpu_options.allow_growth = True #pylint: disable=E1101
    tf.Session(config=config).__enter__()

    #begin mujoco style
    def make_env_fn(rank):
        def _thunk():
            env = make_env(env_id, earlyTerminationTime_ms, rank, seed)
            return env
        return _thunk
    set_global_seeds(seed)
    #end mujoco style

    envs = [make_env_fn(i) for i in range(nenv)]
    env = SubprocVecEnv(envs)
    env = VecNormalize(env, ret=False)

    policy = {'MlpPolicy4' : MlpPolicy4, 'MlpPolicy3' : MlpPolicy3, 'MlpPolicy5': MlpPolicy5}[policy]

    if anneal_ent_coef==0:
        entropy_coef = ent_coef
    elif anneal_ent_coef==1:
        entropy_coef = lambda f : f*ent_coef

    learn(policy=policy, env=env, nsteps=128, nminibatches=4,
        lam=0.95, gamma=0.99, noptepochs=3, log_interval=10,
        ent_coef=entropy_coef,
        lr=lambda f : f * initial_lr,
        cliprange=lambda f : f * 0.1,
        total_timesteps=int(num_timesteps * 1.1),
        save_interval=500, loadModel=loadModel,
          gpu=gpu)

def str2bool(v):
    import argparse
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    parser = iarc_arg_parser()
    parser.add_argument('--nenv', help='Number of environments to run',type = int, default=int(5))
    parser.add_argument('--gpu', help='which gpu to run on', choices=[0, 1], type=int, default=int(0))
    parser.add_argument('--anneal_ent_coef', help='Do you want to anneal entropy coefficient?', choices=[0, 1], type=int, default=int(0))
    args = parser.parse_args()



    # set up data logging directory!
    if (not os.path.isdir(args.logdir)):
        os.makedirs(args.logdir)
    test_n = len(list(n for n in os.listdir(args.logdir) if n.startswith('test')))
    this_test = args.logdir + "/test" + str(test_n + 1)
    os.makedirs(this_test)
    logger.configure(this_test, ['tensorboard'])

    import time
    seed = int(time.time())

    train(args.env, num_timesteps=args.num_timesteps, seed=seed,
          policy=args.policy, earlyTerminationTime_ms=args.earlyTermT_ms,
          loadModel=args.model, nenv=args.nenv,
          ent_coef=args.ent_coef, initial_lr=args.initial_lr,
          gpu=args.gpu, anneal_ent_coef=args.anneal_ent_coef) #, logdir=args.logdir, render=args.render,
          # newModel=args.newModel, earlyTermT_ms=args.earlyTermT_ms)

if __name__ == '__main__':
    main()
