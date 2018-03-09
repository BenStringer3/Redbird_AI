from baselines import logger


def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    import argparse
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def str2bool(v):
    import argparse
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def iarc_arg_parser():
    """
    Create an argparse.ArgumentParser for iarc environments
    """
    parser = arg_parser()
    parser.add_argument('--render', help='To render or not to render (0 or 1)', type=str2bool, default=False)
    parser.add_argument('--policy', help='Policy architecture', choices=['MlpPolicy3', 'MlpPolicy4', 'MlpPolicy5'], default='MlpPolicy3')
    parser.add_argument('--env', help='environment ID', default='IARC_Game_Board-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(10e7))
    parser.add_argument('--logdir', help='path to logging directory', default='/tmp/redbird_AI_logdir/')
    parser.add_argument('--model', help='Model path', default=None)
    parser.add_argument('--initial_lr', help='Initial learning rate', type = float, default=float(2.5e-4))
    parser.add_argument('--ent_coef', help='entropy coefficient', type=float, default=0.01)
    parser.add_argument('--earlyTermT_ms', help='time in ms to cut the game short at', type=int, default=10*60*1000)
    return parser

def save_model(update_num, ob_rms, ret_rms):
    import os.path as osp
    import os
    import tensorflow as tf
    import pickle

    checkdir = osp.join(logger.get_dir(), 'checkpoints')
    os.makedirs(checkdir, exist_ok=True)
    savepath = osp.join(checkdir, '%.5i.ckpt' % update_num)
    print('Saving to', savepath)
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    saver = tf.train.Saver(var_list=tf.global_variables())
    saver.save(tf.get_default_session(), savepath)
    with open(osp.join(checkdir, '%.5i.pik' % update_num), 'wb') as f:
        pickle.dump([ob_rms, ret_rms], f, -1)

def load_model(modelPath):
    import tensorflow as tf
    import pickle

    print('loading old model')
    var_list = tf.global_variables()
    for vars in var_list:
        try:
            saver = tf.train.Saver({vars.name[:-2]: vars})  # the [:-2] is kinda jerry-rigged but ..
            saver.restore(tf.get_default_session(), modelPath + '.ckpt')
            print("found " + vars.name)
        except:
            print("couldn't find " + vars.name)

    with open(modelPath + '.pik', 'rb') as f:
        ob_rms, ret_rms = pickle.load(f)
    print('found observation scaling')


    print('finished loading model')

    return ob_rms, ret_rms

def make_env(env_id, earlyTerminationTime_ms, rank, seed):
    import gym
    from baselines.bench import Monitor
    from Redbird_AI.common.rb_monitor import RB_Monitor
    import os

    env = gym.make(env_id)
    env.seed(seed + 1000 * rank)
    env.env.earlyTerminationTime_ms = earlyTerminationTime_ms
    # env = bench.Monitor(env, logger.get_dir())
    env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
    env = RB_Monitor(env)
    return env