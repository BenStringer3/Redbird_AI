import baselines.common.tf_util as U
import tensorflow as tf
# import redbird_policy
from baselines.common import set_global_seeds
from Redbird_AI.common.policies import MlpPolicy3
import gym
import numpy as np

def demo(env_id, seed, kind, logdir, numRuns, loadModel, earlyTermT_ms=None):
    sess = U.single_threaded_session() #tensorflow session
    sess.__enter__()

    set_global_seeds(seed)
    env = gym.make(env_id)
    # def policy_fn(name, ob_space, ac_space): # pylint: disable=W0613
    #     return redbird_policy.RedbirdPolicy(name=name, ob_space=ob_space, ac_space=ac_space, kind=kind)

    env.seed(seed)
    if earlyTermT_ms is not None:
        env.env.earlyTerminationTime_ms = earlyTermT_ms

    done = False
    stochastic = False
    ob_space = env.observation_space
    ac_space = env.action_space
    ob_shape = [None] + list(ob_space.shape)
    ob = U.get_placeholder("X", tf.float32, ob_shape)
    try:
        nact = np.sum(ac_space.nvec)
    except:
        nact = ac_space.shape[0] * 2
    model = MlpPolicy3(ob, tf.get_default_session(), nact, ac_space, reuse=False)
    # oldpi = policy_fn("oldpi", ob_space, ac_space)  # Construct network for new policy

    # ob = U.get_placeholder_cached(name="ob")
    # ac = pi.pdtype.sample_placeholder([None])

    U.initialize()

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

    # for i in range(numRuns):
    #     ob = env.reset()
    #     done = False
    #     while not done:
    #         ac, vpred = oldpi.act(stochastic, ob)
    #         ob, rew, done, info = env.step(ac)
    #         env.render()
    #
    #         if earlyTermT_ms is not None and info["time_ms"] >= earlyTermT_ms:
    #             done = True
    #
    # env.close()
    for i in range(numRuns):
        ob = env.reset()
        done = False
        states= None
        while not done:
            ob = np.expand_dims(ob, 0)
            actions, values, states, neglogpacs = model.step(ob, states, done)
            ob, rew, done, info = env.step(actions)
            env.render()

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
    parser.add_argument('--kind', help='type of network (small, large, dense)', default='dense')
    parser.add_argument('--logdir', help='path to logging directory', default='/tmp/redbird_AI_logdir/')
    parser.add_argument('--numRuns', help='number of times to run the sim', type=int, default=7)
    parser.add_argument('--earlyTermT_ms', help='time in ms to cut the game short at', type=int, default=10*60*1000)
    parser.add_argument('--model', help='model to load', default=None)
    args = parser.parse_args()
    print("beginning demo")
    demo(env_id=args.env, seed=args.seed, kind=args.kind, logdir=args.logdir, numRuns=args.numRuns, loadModel=args.model, earlyTermT_ms=args.earlyTermT_ms)

if __name__ == '__main__':
    main()