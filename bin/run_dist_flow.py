import os
import sys
import argparse
import time
import numpy as np

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import multiagent.scenarios as scenarios

from multiagent.environment import DistflowEnv


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-s', '--scenario', default='dist_flow.py', help='Path of the scenario python script.')
    parser.add_argument('-n', '--n_agent', type=int, default=2, help='Set the number of agents, both good agents and adversaries')
    parser.add_argument('-w', '--width', type=int, default=6, help='Width of the world.')
    parser.add_argument('-he', '--height', type=int, default=6, help='Height of the world.')
    parser.add_argument('-l', '--length', type=int, default=800, help='Define the minimum step size.')
    parser.add_argument('-r', '--render', action='store_true', help='Render or not, default is not.')

    args = parser.parse_args()

    scenario = scenarios.load(args.scenario).Scenario()
    world = scenario.make_world(num_agents=args.n_agent, width=args.width, height=args.height, unit=np.array([10, 10]), max_steps=args.length)

    env = DistflowEnv(world, scenario.reset_world, scenario.reward, scenario.observation, match_callback=scenario.match, update_callback=scenario.update, info_callback=None, shared_viewer=True)

    if args.render:
        env.render(mode='rgb_array')
    # policies = [RandomPolicy(env, i) for i in range(env.n)]

    obs_n = env.reset()
    step, max_steps = 0, args.length - 1

    while step < max_steps:

        # random sample landmarks
        landmarks = np.random.choice(world.landmarks, size=10)

        obs_n, reward_n, done_n, _ = env.step(landmarks)

        if args.render:
            env.render(mode='rgb_array')
            time.sleep(0.1)

        step += 1

    env.close()
