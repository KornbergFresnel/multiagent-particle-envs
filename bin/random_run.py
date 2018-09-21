import os
import sys
import argparse

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import multiagent.scenarios as scenarios

from multiagent.environment import MultiAgentEnv
from multiagent.policy import RandomPolicy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-s', '--scenario', default='push_ball.py', help='Path of the scenario python script.')
    parser.add_argument('-n', '--n_agent', type=int, default=2, help='Set the number of agents, both good agents and adversaries')
    parser.add_argument('-m', '--multi_viewer', action='store_false', help='Create multiple viewers or not.')
    parser.add_argument('-l', '--length', type=int, default=800, help='Define the minimum step size.')
    args = parser.parse_args()

    scenario = scenarios.load(args.scenario).Scenario()
    world = scenario.make_world(num_agents=args.n_agent)

    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None,
                        shared_viewer=args.multi_viewer)
    env.render()
    policies = [RandomPolicy(env, i) for i in range(env.n)]

    obs_n = env.reset()
    step, min_steps = 0, args.length

    while step < min_steps:
        act_n = []

        for i, policy in enumerate(policies):
            act_n.append(policy.action(obs_n[i]))

        obs_n, reward_n, done_n, _ = env.step(act_n)
        env.render()

        step += 1
    
    env.close()
