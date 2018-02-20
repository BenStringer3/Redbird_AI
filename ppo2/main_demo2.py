#!/usr/bin/env python3
import sys
from baselines import logger
# from baselines.common.cmd_util import make_atari_env, atari_arg_parser
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from redbird_ppo2 import learn, Runner, Model
from policies import CnnPolicy, LstmPolicy, LnLstmPolicy, MlpPolicy3
import multiprocessing
import tensorflow as tf
import gym
from baselines.common import set_global_seeds


def demo(*, policy, env, nsteps, loadModel=None, render=False):
    ob_space = env.observation_space
    ac_space = env.action_space

    make_model = lambda: Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=1, nbatch_train=nsteps,
                               nsteps=nsteps, ent_coef=0.01, vf_coef=0.5,
                               max_grad_norm=0.5)

    model = make_model()
    if loadModel is not None:
        model.load(loadModel)
    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=0.99, lam=0.95, demo=True)

    for update in range(1, 100):
        obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run()  # pylint: disable=E0632

    env.close()

def train(env_id, num_timesteps, seed, policy, earlyTerminationTime_ms, loadModel, render):

    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    config.gpu_options.allow_growth = True #pylint: disable=E1101
    tf.Session(config=config).__enter__()

    env = gym.make(env_id)
    env.seed(seed)
    env.env.earlyTerminationTime_ms = earlyTerminationTime_ms
    set_global_seeds(seed)
    policy = {'cnn' : CnnPolicy, 'lstm' : LstmPolicy, 'lnlstm' : LnLstmPolicy, 'mlp' : MlpPolicy3}[policy]
    demo(policy=policy, env=env, nsteps=128, loadModel=loadModel,render=render)

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
    parser.add_argument('--model', help='Model path', default=None)
    parser.add_argument('--earlyTermT_ms', help='time in ms to cut the game short at', type=int, default=10*60*1000)
    args = parser.parse_args()


    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed,
          policy=args.policy, earlyTerminationTime_ms=args.earlyTermT_ms, loadModel=args.model,render=True) #, logdir=args.logdir, render=args.render,
          # newModel=args.newModel, earlyTermT_ms=args.earlyTermT_ms)

if __name__ == '__main__':
    main()
