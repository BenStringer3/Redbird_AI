#!/usr/bin/env python3
import sys
sys.path.append('/home/redbird_general/Desktop/Redbird_AI2/')

from baselines import logger
# from baselines.common.cmd_util import make_atari_env, atari_arg_parser
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from redbird_ppo2 import learn, Runner #, Model
from Redbird_AI.ppo2.corvus import Corvus
from Redbird_AI.common.policies import  MlpPolicy3, LstmPolicy, LstmPolicy3
import multiprocessing
import tensorflow as tf
import gym
from baselines.common import set_global_seeds
from Redbird_AI.common.cmd_util import iarc_arg_parser, load_model
import time

def demo(*, policy, env, nsteps, loadModel=None, render=False, gpu=0):
    ob_space = env.observation_space
    ac_space = env.action_space

    with tf.device('/device:GPU:' + str(gpu)):
        make_model = lambda: Corvus(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=1, nbatch_train=nsteps,
                               nsteps=nsteps,  vf_coef=0.5,
                               max_grad_norm=0.5, gpu=gpu, nenvs=1)

        model = make_model()
    if loadModel is not None:
        ob_rms, env.ret_rms = load_model(loadModel)
        try:
            if env.ob_rms.mean.shape == ob_rms.mean.shape:
                env.ob_rms = ob_rms
            else:
                print("couldn't isntall observation scaling")
        except:
            print("couldn't isntall observation scaling")
        #load(loadModel)
        # import pickle
        # try:
        #     with open(loadModel + '.pik', 'rb') as f:
        #         env.ob_rms, env.ret_rms = pickle.load(f)
        #     print('found observation scaling')
        # except:
        #     print('could not find observation scaling')

    # runner = Runner(env=env, model=model, nsteps=nsteps, gamma=0.99, lam=0.95, demo=True)

    for i in range(25):
        ob = env.reset()
        done = [False]
        states= model.policy_model.initial_state
        while not any(done):
            # ob = np.expand_dims(ob, 0)
            actions, values, states, neglogpacs = model.step(ob, states, done)
            ob, rew, done, info = env.step(actions)
            # env.render()


    env.close()
    # for update in range(1, 100):
    #     obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run()  # pylint: disable=E0632
    #
    # env.close()

def train(env_id, num_timesteps, seed, policy, earlyTerminationTime_ms, loadModel, render, gpu=0):
    from baselines.bench import Monitor
    import os
    from baselines.common.vec_env.vec_normalize import VecNormalize
    from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    config.gpu_options.allow_growth = True #pylint: disable=E1101
    tf.Session(config=config).__enter__()

    # env = gym.make(env_id)
    # env.seed(seed)
    # env.env.earlyTerminationTime_ms = earlyTerminationTime_ms
    # # env = Monitor(env, logger.get_dir())
    # set_global_seeds(seed)
    # env = VecNormalize(env)


    #begin mujoco style
    def make_env():

        from baselines import bench
        env = gym.make(env_id)
        env.seed(seed)
        env.env.earlyTerminationTime_ms = earlyTerminationTime_ms
        env.env.render_bool = True
        # env = bench.Monitor(env, logger.get_dir())
        # env = Monitor(env, logger.get_dir())
        return env

    env = DummyVecEnv([make_env])
    env.num_envs = 1
    env = VecNormalize(env, ret=True, ob=False)
    set_global_seeds(seed)
    #end mujoco style

    # envs = [make_env(i) for i in range(20)]
    # env = SubprocVecEnv(envs)
    # env = VecNormalize(env)

    policy = {'MlpPolicy3' : MlpPolicy3, 'LstmPolicy':LstmPolicy, 'LstmPolicy3':LstmPolicy3}[policy]
    demo(policy=policy, env=env, nsteps=128, loadModel=loadModel,render=render, gpu=gpu)



def main():
    parser = iarc_arg_parser()
    parser.add_argument('--gpu', help='which gpu to run on', choices=[0, 1], type=int, default=int(0))

    args = parser.parse_args()

    logger.configure('/tmp/main_demo2_logdir/')

    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed,
          policy=args.policy, earlyTerminationTime_ms=args.earlyTermT_ms, loadModel=args.model,render=True, gpu=args.gpu) #, logdir=args.logdir, render=args.render,
          # newModel=args.newModel, earlyTermT_ms=args.earlyTermT_ms)

if __name__ == '__main__':
    main()
