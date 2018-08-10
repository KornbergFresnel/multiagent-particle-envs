import os
import sys
import argparse

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from multiagent.environment import MultiAgentEnv
from multiagent.policy import InteractivePolicy
import multiagent.scenarios as scenarios


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_agent', type=int, default=2, help='Set the number of agents')
    parser.add_argument('--n_adv', type=int, default=0, help='Define how many adversary agents among the total agents')
    parser.add_argument('--comm', action='store_true', help='Use communication action or not.')
    parser.add_argument('--train', action='store_true', help='Train or not')

    args = parser.parse_args()




