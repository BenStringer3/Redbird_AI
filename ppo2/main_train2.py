#!/usr/bin/env python3
import sys
sys.path.append('/home/redbird_general/Desktop/Redbird_AI2/')

from baselines import logger
from Redbird_AI.ppo2.redbird_ppo2 import learn
from Redbird_AI.common.policies import  MlpPolicy3, MlpPolicy4, LstmPolicy3, LstmPolicy
import multiprocessing
import tensorflow as tf
from baselines.common import set_global_seeds
from Redbird_AI.common.cmd_util import iarc_arg_parser, make_env, save_model
import os
from time import sleep

def train(env_id, num_timesteps, seed, policy, earlyTerminationTime_ms, loadModel, nenv, ent_coef, initial_lr=2.5e-4, gpu=0, anneal_ent_coef=0, debug=False):
    from baselines.common.vec_env.vec_normalize import VecNormalize
    from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    config.gpu_options.allow_growth = True #pylint: disable=E1101
    if debug:
        from tensorflow.python import debug as tf_debug
        tf_debug.LocalCLIDebugWrapperSession(tf.Session(config=config).__enter__())
    else:
        tf.Session(config=config).__enter__()

    #begin mujoco style
    def make_env_fn(rank, env_id):
        sleep(0.01) #TODO figure out segmentation fault issue
        def _thunk():
            env = make_env(env_id, earlyTerminationTime_ms, rank, seed)
            return env
        return _thunk
    set_global_seeds(seed)
    #end mujoco style

    envs = [make_env_fn(i, env_id) for i in range(nenv)]
    env = SubprocVecEnv(envs)
    env = VecNormalize(env, ret=True, ob=False)

    policy = {'MlpPolicy4' : MlpPolicy4, 'MlpPolicy3' : MlpPolicy3, 'LstmPolicy3': LstmPolicy3, 'LstmPolicy': LstmPolicy}[policy]

    if anneal_ent_coef==0:
        entropy_coef = ent_coef
    elif anneal_ent_coef==1:
        entropy_coef = lambda f : f*ent_coef

    run_info = "policy=" + policy.__name__ + ", nenvs=" + str(nenv) + ", env_id=" + env_id + ", earlyterm=" + str(earlyTerminationTime_ms) + "initial_lr=" + str(initial_lr)

    try:
        learn(policy=policy, env=env, nsteps=128, nminibatches=10,
            lam=0.95, gamma=0.99, noptepochs=4, log_interval=10,
            ent_coef=entropy_coef,
            lr=lambda f : f * initial_lr,
            cliprange=lambda f : f * 0.1,
            total_timesteps=int(num_timesteps * 1.1),
            save_interval=1000, loadModel=loadModel,
              gpu=gpu, run_info=run_info)
    except KeyboardInterrupt:
        print('keyboard interrupt triggered. Attempting clean exit')
        save_model('final', env.ob_rms, env.ret_rms)
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

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
