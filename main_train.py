
from mpi4py import MPI #parallelization stuff
from datetime import datetime #for seeding random number generator
import gym
import redbird_policy
from baselines.common import set_global_seeds
import redbird_pposgd
import os

def train(env_id, num_timesteps, seed, kind, logdir, render, newModel):
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
    def policy_fn(name, ob_space, ac_space): # pylint: disable=W0613
        return redbird_policy.RedbirdPolicy(name=name, ob_space=ob_space, ac_space=ac_space, kind=kind)

    env.seed(workerseed)

    redbird = redbird_pposgd.RedbirdPposgd(rank, this_test, last_test)

    redbird.learn(env, policy_fn,
           max_timesteps=int(num_timesteps * 1.1),
           timesteps_per_actorbatch=128,  # 256,
           clip_param=0.2, entcoeff=0.01,
           optim_epochs=3, optim_stepsize=2.5e-4, optim_batchsize=32,
           gamma=0.99, lam=0.95,
           schedule='linear',
           render=render, newModel=newModel
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
    parser.add_argument('--newModel', help='Create new model or use most recently created', type=str2bool, default=True)
    args = parser.parse_args()
    print("beginning training")
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed, kind=args.kind, logdir=args.logdir, render=args.render, newModel=args.newModel)


if __name__ == '__main__':
    main()