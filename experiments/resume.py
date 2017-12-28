"""
Fine tune a trained policy/qf
"""
import argparse

import joblib

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str,
                        help='Path to snapshot file to fine tune.')
    args = parser.parse_args()

    data = joblib.load(args.path)
    algo = data['algorithm']
    algo.train()
