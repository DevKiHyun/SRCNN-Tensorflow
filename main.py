import argparse
import sys

sys.path.append('..')
import SRCNN.srcnn as srcnn
import SRCNN.train as train

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='-')
    parser.add_argument('--beta_1', type=float, default=0.9, help='-')
    parser.add_argument('--beta_2', type=float, default=0.999, help='-')
    parser.add_argument('--epsilon', type=float, default=1e-08, help='-')
    parser.add_argument('--training_epoch', type=int, default=1500, help='-')
    parser.add_argument('--batch_size', type=int, default=64, help='-')
    parser.add_argument('--n_channel', type=int, default=1, help='-')
    args, unknown = parser.parse_known_args()

    SRCNN = srcnn.SRCNN(args)
    train.training(SRCNN, args)
