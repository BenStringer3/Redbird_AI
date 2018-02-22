#!/usr/bin/env python3
import sys
from baselines import logger
# from baselines.common.cmd_util import make_atari_env, atari_arg_parser
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from redbird_ppo2 import learn
from policies import CnnPolicy, LstmPolicy, LnLstmPolicy, MlpPolicy3
import multiprocessing
import tensorflow as tf
import gym
from baselines.common import set_global_seeds
from baselines.bench import Monitor
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
            env.seed(seed + rank)
            env.env.earlyTerminationTime_ms = earlyTerminationTime_ms
            env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
            # return wrap_deepmind(env, **wrapper_kwargs)
            return env
        return _thunk
    set_global_seeds(seed)
    envs = [make_env(i + start_index) for i in range(num_env)]
    ret= SubprocVecEnv(envs)
    return ret

def train(env_id, num_timesteps, seed, policy, earlyTerminationTime_ms, loadModel):
    from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
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
    def make_env(rank):
        def _thunk():
            from baselines import bench
            env = gym.make(env_id)
            env.seed(seed + rank)
            env.env.earlyTerminationTime_ms = earlyTerminationTime_ms
            # env = bench.Monitor(env, logger.get_dir())
            env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
            return env
        return _thunk
    # env = DummyVecEnv([make_env])
    # env = VecNormalize(env)
    set_global_seeds(seed)
    #end mujoco style

    envs = [make_env(i) for i in range(32)]
    env = SubprocVecEnv(envs)
    env = VecNormalize(env)

    policy = {'cnn' : CnnPolicy, 'lstm' : LstmPolicy, 'lnlstm' : LnLstmPolicy, 'mlp' : MlpPolicy3}[policy]

    # env = VecFrameStack(make_IARC_env(env_id, 8, seed, earlyTerminationTime_ms), 4)

    learn(policy=policy, env=env, nsteps=128, nminibatches=4,
        lam=0.95, gamma=0.99, noptepochs=3, log_interval=10,
        ent_coef=.01,
        lr=lambda f : f * 2.5e-4,
        cliprange=lambda f : f * 0.1,
        total_timesteps=int(num_timesteps * 1.1),
        save_interval=500, loadModel=loadModel)

def str2bool(v):
    import argparse
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    import argparse
    import os
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm', 'mlp'], default='mlp')
    parser.add_argument('--env', help='environment ID', default='IARC_Game_Board-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(10e7))
    parser.add_argument('--logdir', help='path to logging directory', default='/tmp/redbird_AI_logdir/')
    parser.add_argument('--model', help='Model path', default=None)
    parser.add_argument('--earlyTermT_ms', help='time in ms to cut the game short at', type=int, default=10*60*1000)
    args = parser.parse_args()

    # set up data logging directory!
    if (not os.path.isdir(args.logdir)):
        os.makedirs(args.logdir)
    test_n = len(list(n for n in os.listdir(args.logdir) if n.startswith('test')))
    this_test = args.logdir + "/test" + str(test_n + 1)
    os.makedirs(this_test)
    logger.configure(this_test, ['tensorboard'])

    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed,
          policy=args.policy, earlyTerminationTime_ms=args.earlyTermT_ms, loadModel=args.model) #, logdir=args.logdir, render=args.render,
          # newModel=args.newModel, earlyTermT_ms=args.earlyTermT_ms)

if __name__ == '__main__':
    main()
