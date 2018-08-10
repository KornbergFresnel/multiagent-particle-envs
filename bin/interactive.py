#!/usr/bin/env python
import os
import sys
import argparse

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import multiagent.scenarios as scenarios

from multiagent.environment import MultiAgentEnv
from multiagent.policy import InteractivePolicy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-s', '--scenario', default='simple_push.py', help='Path of the scenario Python script.')
    parser.add_argument('--n_agent', type=int, default=2, help='Set the number of agents')
    parser.add_argument('--n_adv', type=int, default=0, help='Define how many adversary agents among the total agents')
    parser.add_argument('--comm', action='store_true', help='Use communication action or not.')
    parser.add_argument('--train', action='store_true', help='Train or not')
    args = parser.parse_args()

    scenario = scenarios.load(args.scenario).Scenario()  # load scenario from script
    world = scenario.make_world()  # create world instant
    # create multi-agent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None,
                        shared_viewer=False)
    # render call to create viewer window (necessary only for interactive policies)
    env.render()
    # create interactive policies for each agent
    policies = [InteractivePolicy(env, i) for i in range(env.n)]
    # execution loop
    obs_n = env.reset()
    while True:
        # query for action from each agent's policy
        act_n = []
        for i, policy in enumerate(policies):
            act_n.append(policy.action(obs_n[i]))
        # step environment
        obs_n, reward_n, done_n, _ = env.step(act_n)
        # render all agent views
        env.render()
        # display rewards
        # for agent in env.world.agents:
        #    print(agent.name + " reward: %0.3f" % env._get_reward(agent))
