
from mpi4py import MPI #parallelization stuff
from datetime import datetime #for seeding random number generator
import gym
# import redbird_policy
from baselines.common import set_global_seeds
# import redbird_pposgd
import os
from baselines import logger
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv


def train(env_id, num_timesteps, seed, kind, logdir, render, loadModel, earlyTermT_ms, initial_lr=2.5e-4):
    import baselines.common.tf_util as U
    rank = MPI.COMM_WORLD.Get_rank() #the id of this process
    sess = U.single_threaded_session() #tensorflow session
    sess.__enter__()

    #set up data logging directory!
    if rank == 0: #if this is the first process
        # set up data logging directory!
        if (not os.path.isdir(logdir)):
            os.makedirs(logdir)
        test_n = len(list(n for n in os.listdir(logdir) if n.startswith('test')))
        this_test = logdir + "/test" + str(test_n + 1)
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
    logger.configure(this_test + '/rank_' + str(rank), ['tensorboard'])

    # workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    # set_global_seeds(workerseed)
    # env = gym.make(env_id)
    # env = VecNormalize(env)

    #begin mujoco style
    def make_env():

        from baselines import bench
        env = gym.make(env_id)
        env.seed(seed+ 10000*rank)
        env.env.earlyTerminationTime_ms = earlyTermT_ms
        env = bench.Monitor(env, logger.get_dir())
        # env = Monitor(env, logger.get_dir())
        return env

    env = DummyVecEnv([make_env])
    env.num_envs = 1
    env = VecNormalize(env)
    set_global_seeds(seed)
    #end mujoco style

    def policy_fn(name, ob_space, ac_space, reuse): # pylint: disable=W0613
        from Redbird_AI.ppo2.policies import MlpPolicy3
        import tensorflow as tf
        return MlpPolicy3(tf.get_default_session(), ob_space, ac_space, [None], 1, reuse)
        # return redbird_policy.RedbirdPolicy(name=name, ob_space=ob_space, ac_space=ac_space, kind=kind)

    # env.seed(workerseed)
    from Redbird_AI.ppo1.redbird_pposgd import RedbirdPposgd
    redbird = RedbirdPposgd(rank, this_test, last_test, earlyTermT_ms=earlyTermT_ms)

    redbird.learn(env, policy_fn,
           max_timesteps=int(num_timesteps * 1.1),
           timesteps_per_actorbatch=128,  # 256,
           clip_param=0.2, entcoeff=0.001, vf_coef=0.5,
           optim_epochs=3, optim_stepsize=initial_lr, optim_batchsize=32,
           gamma=0.99, lam=0.95,
           schedule='linear',
           render=render, loadModel=loadModel, lr=lambda f : f * 2.5e-4
           )
    env.close()

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
    parser.add_argument('--env', help='environment ID', default='IARC_Game_Board-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(10e7))
    parser.add_argument('--kind', help='type of network (small, large, dense)', default='dense')
    parser.add_argument('--logdir', help='path to logging directory', default='/tmp/redbird_AI_logdir/')
    parser.add_argument('--render', help='To render or not to render (0 or 1)', type=str2bool, default=False)
    parser.add_argument('--model', help='Create new model or use most recently created',  default=None)
    parser.add_argument('--earlyTermT_ms', help='time in ms to cut the game short at', type=int, default=10*60*1000)
    parser.add_argument('--initial_lr', help='Initial learning rate', type = float, default=float(2.5e-4))

    args = parser.parse_args()


    print("beginning training")
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed, kind=args.kind, logdir=args.logdir, render=args.render, loadModel=args.model, earlyTermT_ms=args.earlyTermT_ms, initial_lr=args.initial_lr)


if __name__ == '__main__':
    main()