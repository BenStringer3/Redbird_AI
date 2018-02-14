

from ppo2 import Redbird_Pposgd2
from policies import CnnPolicy, LstmPolicy, LnLstmPolicy, MlpPolicy2
import tensorflow as tf
import gym
import os
from baselines.common import set_global_seeds
from mpi4py import MPI #parallelization stuff
# from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

# def make_IARC_env(env_id, num_env, seed, wrapper_kwargs=None, start_index=0):
#     """
#     Create a wrapped, monitored SubprocVecEnv for Atari.
#     """
#     if wrapper_kwargs is None: wrapper_kwargs = {}
#     def make_env(rank): # pylint: disable=C0111
#         def _thunk():
#             env = gym.make(env_id)
#             env.seed(seed + rank)
#             # env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
#             # return wrap_deepmind(env, **wrapper_kwargs)
#             return env
#         return _thunk
#     set_global_seeds(seed)
#     return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])

def train(env_id, num_timesteps, seed, policy, logdir, render, newModel, earlyTermT_ms):
    import baselines.common.tf_util as U
    rank = MPI.COMM_WORLD.Get_rank() #the id of this process
    sess = U.single_threaded_session() #tensorflow session
    sess.__enter__()


    #set up data logging directory!
    if rank == 0: #if this is the first process
        if (not os.path.isdir(logdir)):
            os.makedirs(logdir)
        test_n = len(list(n for n in os.listdir(logdir) if n.startswith('test')))
        this_test = logdir + "/test" + str(test_n + 1)
        last_test =  logdir + "/test" + str(test_n)
        os.makedirs(this_test)
        for i in range(1, MPI.COMM_WORLD.Get_size()): # tell the other processes which test directory we're in
            MPI.COMM_WORLD.send(test_n+1, dest=i, tag=11)
        os.makedirs(this_test + '/rank_' + str(rank))
    else:
        test_n = MPI.COMM_WORLD.recv(source=0, tag=11) #receive test_n from rank 0 process
        this_test = logdir + "/test" + str(test_n)
        if test_n > 0:
            last_test = logdir + "/test" + str(test_n - 1)
        else:
            last_test = None
        os.makedirs(this_test + '/rank_' + str(rank))

    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)
    env = gym.make(env_id)
    env.seed(workerseed)

    redbird = Redbird_Pposgd2(rank, this_test, last_test, render=render, earlyTermT_ms=earlyTermT_ms)


    policy = {'cnn' : CnnPolicy, 'lstm' : LstmPolicy, 'lnlstm' : LnLstmPolicy, 'mlp' : MlpPolicy2}[policy]
    redbird.learn(policy=policy, env=env, nsteps=128, nminibatches=4,
        lam=0.95, gamma=0.99, noptepochs=4, log_interval=1,
        ent_coef=.01,
        lr=lambda f : f * 2.5e-4,
        cliprange=lambda f : f * 0.1,
        total_timesteps=int(num_timesteps * 1.1))


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
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm', 'mlp'], default='mlp')
    parser.add_argument('--env', help='environment ID', default='IARC_Game_Board-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(10e7))
    parser.add_argument('--logdir', help='path to logging directory', default='/tmp/redbird_AI_logdir/')
    parser.add_argument('--render', help='To render or not to render (0 or 1)', type=str2bool, default=False)
    parser.add_argument('--newModel', help='Create new model or use most recently created', type=str2bool, default=True)
    parser.add_argument('--earlyTermT_ms', help='time in ms to cut the game short at', type=int, default=10*60*1000)
    args = parser.parse_args()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed,
          policy=args.policy, logdir=args.logdir, render=args.render,
          newModel=args.newModel, earlyTermT_ms=args.earlyTermT_ms)


if __name__ == '__main__':
    main()
