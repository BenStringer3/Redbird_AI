

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
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm', 'mlp'], default='mlp')
    parser.add_argument('--env', help='environment ID', default='IARC_Game_Board-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(10e7))
    parser.add_argument('--logdir', help='path to logging directory', default='/tmp/redbird_AI_logdir/')
    parser.add_argument('--model', help='Model path', default=None)
    parser.add_argument('--initial_lr', help='Initial learning rate', type = float, default=float(2.5e-4))
    parser.add_argument('--earlyTermT_ms', help='time in ms to cut the game short at', type=int, default=10*60*1000)
    return parser