
from mpi4py import MPI #parallelization stuff
import gym
# import redbird_policy
from baselines.common import set_global_seeds
# import redbird_pposgd
import os
from baselines import logger
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
import sys
sys.path.append('/home/bcstri01/env2/Redbird_AI')
sys.path.append('/home/bcstri01/env2')
from Redbird_AI.common.cmd_util import iarc_arg_parser
from Redbird_AI.ppo1.redbird_pposgd import RedbirdPposgd
from Redbird_AI.common.policies import MlpPolicy3, MlpPolicy4

def train(env_id, num_timesteps, seed, kind, logdir, render, loadModel, earlyTermT_ms, ent_coef, initial_lr=2.5e-4):
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
        # os.makedirs(this_test + '/rank_' + str(rank))
        logger.configure(this_test, ['tensorboard'])

    else:
        test_n = MPI.COMM_WORLD.recv(source=0, tag=11) #receive test_n from rank 0 process
        this_test = logdir + "/test" + str(test_n)
    #     if test_n > 0:
    #         last_test = logdir + "/test" + str(test_n - 1)
    #     else:
    #         last_test = None
    #     os.makedirs(this_test + '/rank_' + str(rank))

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
    env = VecNormalize(env,ret=True)
    set_global_seeds(seed)
    #end mujoco style

    # def policy_fn(name, ob_space, ac_space, reuse): # pylint: disable=W0613
    #     from Redbird_AI.common.policies import MlpPolicy3
    #     import tensorflow as tf
    #     return MlpPolicy3( X, sess, nact,  ac_space, reuse=reuse)#(tf.get_default_session(), ob_space, ac_space, [None], 1, reuse)
    #     # return redbird_policy.RedbirdPolicy(name=name, ob_space=ob_space, ac_space=ac_space, kind=kind)

    # env.seed(workerseed)
    redbird = RedbirdPposgd(rank, this_test, None, earlyTermT_ms=earlyTermT_ms)

    policy = {"MlpPolicy3": MlpPolicy3, "MlpPolicy4" : MlpPolicy4}[kind]
    redbird.learn(env, policy, #policy_fn,
           max_timesteps=int(num_timesteps * 1.1),
           timesteps_per_actorbatch=128,  # 256,
           clip_param=0.2, entcoeff=ent_coef, vf_coef=0.5,
           optim_epochs=3, optim_stepsize=initial_lr, optim_batchsize=32,
           gamma=0.99, lam=0.95,
           schedule='linear',
           render=render, loadModel=loadModel, lr=lambda f : f * 2.5e-4
           )
    env.close()


def main():
    parser = iarc_arg_parser()
    args = parser.parse_args()

    print("beginning training")
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed, kind=args.policy, logdir=args.logdir, render=args.render, loadModel=args.model, earlyTermT_ms=args.earlyTermT_ms, ent_coef=args.ent_coef, initial_lr=args.initial_lr)


if __name__ == '__main__':
    main()