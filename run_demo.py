import argparse
import numpy as np
import os
import sys
import tensorflow as tf
import yaml

from lib.utils import load_graph_data
from model.dcrnn_supervisor import DCRNNSupervisor


def run_dcrnn(args):
    with open(args.config_filename) as f:
        config = yaml.load(f)
    tf_config = tf.ConfigProto()
    if args.use_cpu_only:
        tf_config = tf.ConfigProto(device_count={'GPU': 0})
    tf_config.gpu_options.allow_growth = True
    graph_pkl_filename = config['data']['graph_pkl_filename']
    _, _, adj_mx = load_graph_data(graph_pkl_filename)
    with tf.Session(config=tf_config) as sess:
        if args.input_data is not None:
            assert args.output_data is not None
            import pickle
            with open(args.input_data, 'rb') as f:
                x = pickle.load(f)
            config["data"]["test_batch_size"] = len(x)

        supervisor = DCRNNSupervisor(adj_mx=adj_mx, **config)
        # supervisor.load(sess, config['train']['model_filename']

        if args.model_filename is not None:  # TODO(piyush) remove
            supervisor.load(sess, args.model_filename)
        else:
            supervisor.load(sess, config['train']['model_filename'])

        if args.input_data is not None:
            results = supervisor.evaluate_on_inputdata(sess, x, num_trials=args.num_trials)
            with open(args.output_data, "wb") as f:
                pickle.dump(results, f)
            print(f"Dumped results to {args.output_data}")
        elif args.n_var_bins is not None:
            results = supervisor.evaluate_varbins(
                sess, args.n_var_bins, percentile=args.percentile_variance)
        else:
            outputs = supervisor.evaluate(sess)
            np.savez_compressed(args.output_filename, **outputs)
            print('Predictions saved as {}.'.format(args.output_filename))


if __name__ == '__main__':
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cpu_only', default=False, type=str, help='Whether to run tensorflow on cpu.')
    parser.add_argument('--config_filename', default='data/model/pretrained/METR-LA/config.yaml', type=str,
                        help='Config file for pretrained model.')
    parser.add_argument('--output_filename', default='data/dcrnn_predictions.npz')
    parser.add_argument("--model_filename", type=str, default=None) # TODO(piyush) remove
    parser.add_argument("--n-var-bins", type=int, default=None)
    parser.add_argument("--percentile-variance", action="store_true", default=False)
    parser.add_argument("--input-data", type=str, default=None)
    parser.add_argument("--output-data", type=str, default=None)
    parser.add_argument("--num-trials", type=int, default=1)
    args = parser.parse_args()
    run_dcrnn(args)
