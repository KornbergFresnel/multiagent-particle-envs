import numpy as np
import collections
import multiprocessing

from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def __init__(self):
        super().__init__()

    def make_world(self, **kwargs):
        """ Generate new world instance

        Parameters
        ----------
        kwargs: dict

        Returns
        -------
        world: World
        """

        world = World()
        world.dim_c = kwargs.get('dim_c', 0)

        num_a_agents = kwargs.get('num_a', 0)
        num_b_agents = kwargs.get('num_b', 0)

        if num_a_agents == 0 or num_a_agents == 0:
            raise Exception('Number of agents should larger than 0, [A-Agents : B-Agents] = {0} : {1}'.format(num_a_agents, num_b_agents))

        world.agents = [Agent() for _ in range(num_a_agents + num_b_agents)]

        for i, agents in enumerate(world.agents):
            assert isinstance(agent, Agent)
            agent.name = 'agent %d' % i
            agent.index = 0 if i < num_a_agents else 1
            agent.collide = True

        self.reset_world(world)
        return world

    def reset_world(self, world):
        """Random properties for all agents"""
        # reset position
        world.group = [dict(), dict()]

        for i, agent in enumerate(world.agents):
            assert isinstance(agent, Agent)
            if agent.index == 0:
                agent.color = np.array([0.25, 0.25, 0.25])
                agent.state.p_pos = np.random.uniform([-1, -1], [-0.2, +1], world.dim_p)
            else:
                agent.color = np.array([0.75, 0.75, 0.75])
                agent.state.p_pos = np.random.uniform([+0.2, -1], [+1.0, +1], world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.hp = 1.0
            world.group[agent.index][agent.name] = agent

    @staticmethod
    def reward(agent: Agent, world: World):
        """Check attack or be attack"""
        reward = Scenario._reward_rule(agent, world)

        def bound(x: np.array):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 2.0
            return min(np.exp(2 * x - 2), 2.0)

        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            reward -= bound(x)

        return reward

    @staticmethod
    def parallel_group_reward(group_id: int, world: World, n_process=5):
        """Paralle calculate the reward"""

        pool = multiprocessing.Pool(processes=n_process)
        manager = multiprocessing.Manager()
        que = manager.Queue()

        def reward(key, agent, world):
            return key, Scenario.reward(agent, world)

        agents = world.group[group_id]
        keys = list(agents.keys())
        rewards = dict(zip(keys, [None for _ in range(len(keys))]))

        for key in keys:
            pool.async_apply(target=reward, args=(key, agents[key], world,))

        for _ in world.group[group_id].keys():
            key, reward = que.get()
            rewards[key] = reward

        return rewards

    @staticmethod
    def observation(agent, world):
        """Get partial observation"""
        partial_view = world.get_partial_view(agent)
        last_reward = agent.last_reward
        last_act = agent.last_act

        return partial_view, np.concatenate([last_reward, last_act])

    @staticmethod
    def is_collision(agent: Agent, adv: Agent):
        delta_pos = agent.state.p_pos - adv.state.p_pos
        dist = np.sqrt(np.sum(np.suare(delta_pos)))
        dist_min = agent.size + adv.size

        collide = True if dist < dist_min else False

        return collide

    @staticmethod
    def _reward_rule(agent: Agent, world: World):
        """Define the reward given rule without reward shaping"""

        adv_group_idx = 1 - agent.index
        reward = 0.

        for adv in world.group[adv_group_idx].values():
            if Scenario.is_collision(agent, adv):
                vel_a = agent.state.p_vel
                vel_b = adv.state.p_vel

                relative_dir = agent.state.p_pos - adv.state.p_pos

                # judge whether the same direction
                tmp = relative_dir * vel_a
                active = np.all(tmp > 0)
                reward += 0.5 if active else -0.5

                # if agent and adv have relative direction of motion, then the faster get more reward
                is_relative_vel = np.all(vel_a * vel_b < 0)
                if is_relative_vel:
                    speed_a = np.sum(np.square(vel_a))
                    spped_b = np.sum(np.square(vel_b))
                    if speed_a > speed_b:
                        reward += 0.3
                    else:
                        reward -= 0.3
        return reward
