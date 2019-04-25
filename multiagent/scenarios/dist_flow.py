import os.path as osp
import numpy as np
import random

from copy import copy
from gym import spaces

from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


def number_generator(mean, var=None, dist_type="gaussian"):
    # TODO(ming): Gaussian distribution

    if dist_type == "gaussian":
        n = np.random.normal(mean, var)
    else:
        raise NotImplementedError

    return max(n, 0)


def random_selector(dataset, p=None, size=None):
    seg, remain = size // len(dataset), size % len(dataset)

    res = []

    for _ in range(seg):
        res.extend(random.sample(dataset, len(dataset)))

    res.extend(random.sample(dataset, remain))

    return res


def check_grid(pos, unit, ratio):
    real_pos = pos + 1.

    grid_id = tuple((real_pos / ratio / unit).astype(np.int32))

    return grid_id


def update_info(node):
    raise NotImplementedError


def convert_grid_id_to_pos(source, world, r=(-1, 1)):
    x, y = source

    noise = -world.unit / 2. + np.random.rand(2) * world.unit
    pos = np.array([x * world.unit[0] + world.unit[0] * 0.5, y * world.unit[1] + world.unit[1] * 0.5]) + noise

    pos[0] = np.clip(pos[0], 0, world.unit[0] * world.width)
    pos[1] = np.clip(pos[1], 0, world.unit[1] * world.width)
    pos = pos * world.ratio - 1.

    assert r[0] <= pos[0] <= r[1] and r[0] <= pos[1] <= r[1], "pos is: {}".format(pos)

    return pos


class Node(object):
    def __init__(self, x, y, node_id):
        self._reabable_node_id = (x, y)
        self._node_id = node_id

        self._agents = dict()
        self._landmarks = dict()

    @property
    def observation(self):
        return [self.n_agents, self.n_landmarks]

    @property
    def n_agents(self):
        return len(self._agents)

    @property
    def n_landmarks(self):
        return len(self._landmarks)

    @property
    def node_id(self):
        return self._node_id

    @property
    def agents(self):
        return self._agents

    @property
    def landmarks(self):
        return self._landmarks

    @property
    def human_node_id(self):
        return self._reabable_node_id

    def update_agent(self, remove_ids=None, add_agents=None):
        """Recalled when match distribution"""

        removed, migrates = [], []

        if remove_ids is not None:
            for key in remove_ids:
                agent = self._agents.pop(key)

                assert not agent.is_on_service, "Agent cannot remove since it is on serving!"
                removed.append(key)

        if add_agents is not None:
            for agent in add_agents:
                # TODO(ming): update agent info
                # agent.callback("update_info", (self,))
                self._agents[agent.name] = agent

    def clear_landmarks(self):
        remove = []
        for e in self.landmarks.values():
            remove.append(e.name)

        for name in remove:
            self._landmarks.pop(name)

    def update_landmarks(self, landmarks):
        # check duration first
        # add landmarks
        for landmark in landmarks:
            self._landmarks[landmark.name] = landmark


class Scenario(BaseScenario):
    def make_world(self, **kwargs):
        """Generate a world instance

        :param kwargs: dict, includes {'height', 'width', 'n_agents', 'n_landmarks'}
        :return: world, World, the world instance
        """

        world = World()
        world.vel_decay = False  # close velocity decay
        world.grids = dict()
        world.height, world.width, world.unit = kwargs['height'], kwargs['width'], kwargs['unit']
        world.ratio = np.array((2. / world.width / world.unit[0], 2. / world.height / world.unit[1]))

        for i in range(kwargs['width']):
            for j in range(kwargs['height']):
                world.grids[(i, j)] = Node(i, j, i * kwargs['width'] + j)

        self.landmark_s_dist = [None for _ in range(kwargs['max_steps'])]
        self.landmark_t_dist = [None for _ in range(kwargs['max_steps'])]
        self.agent_n_dist = [(50., 1.) for _ in range(kwargs['max_steps'])]
        self.landmark_n_dist = [(50., 1.) for _ in range(kwargs['max_steps'])]

        # add agents
        self.reset_world(world)
        self.time = 0

        return world

    def update_agent_domain(self, world, n_agents=0):
        """ Update agent domain, also its status """
        # TODO(ming): need add callback function for each new agents, then collect ava_agents
        n_agents = int(n_agents)

        if n_agents > 0:
            world.agents = [Agent() for _ in range(n_agents)]
            s_grids = random_selector(list(world.grids.keys()), p=self.landmark_s_dist[self.time], size=n_agents)

            for i, (agent, source) in enumerate(zip(world.agents, s_grids)):
                agent.name = 'agent_%d_%d' % (i, self.time)
                agent.collide = False
                agent.silent = True
                agent.landmark = None
                agent.is_on_service = False
                agent.grid_id = source
                agent.state.p_pos = convert_grid_id_to_pos(source, world, (-1, 1))
                agent.state.p_vel = np.zeros(world.dim_p)
                agent.state.c = np.zeros(world.dim_c)
                agent.color = np.array([0.75, 0.75, 0.75])
                agent.size *= 0.25

                # agent.bind_callback("update_info", update_info)
                agent.action_space = spaces.Discrete(world.dim_p * 2 + 1)

                world.grids[source].update_agent(add_agents=[agent])

        for agent in world.agents:
            # check domain
            if getattr(agent, "grid_id", None) is not None:
                agent.grid_id = check_grid(agent.state.p_pos, world.unit, world.ratio)
                assert agent.grid_id[0] < world.width and agent.grid_id[1] < world.height, "grid_id infer error: {} {} {} {}".format(agent.grid_id, agent.state.p_pos, world.unit, world.ratio)
            else:
                source = agent.grid_id
                world.grids[source].update_agent(remove_ids=[agent.name])
                agent.grid_id = random_selector(list(world.grids.keys()), p=None, size=1)

            world.grids[agent.grid_id].update_agent(add_agents=[agent])

            if agent.landmark is not None and agent.landmark.target_grid_id == agent.grid_id:
                # is agent is on its target domain, stop it
                agent.state.p_vel -= agent.state.p_vel
            else:
                pass
                # # random_stop() or random_start()
                # if not agent.is_on_service:
                #     # TODO(ming): random stop
                #     raise NotImplementedError
                # else:
                #     # random_migrate() avoid
                #     pass

        # for node in world.grids.values():  # since agent changed their domain, grid needs to remove these agents
        #     node.update_agent(add_agents)

    def update_landmarks(self, world, n_landmarks=0):
        """ Remove dead landmarks and add new landmarks """

        for grid in world.grids.values():
            grid.clear_landmarks()

        is_all_off = True

        for agent in world.agents:
            if agent.is_on_service:
                is_all_off = False
                break

        if not is_all_off:
            # print("all off")
            return

        n_landmarks = int(n_landmarks)
        landmarks = [Landmark() for _ in range(n_landmarks)]

        s_grids = random_selector(list(world.grids.keys()), p=self.landmark_s_dist[self.time], size=n_landmarks)
        t_grids = random_selector(list(world.grids.keys()), p=self.landmark_t_dist[self.time], size=n_landmarks)

        for i, (landmark, source, target) in enumerate(zip(landmarks, s_grids, t_grids)):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.grid_id = source
            landmark.target_grid_id = target
            landmark.done = False
            landmark.color = np.array([229 / 255, 132 / 255, 129 / 255])
            landmark.size *= 0.2

            landmark.state.p_pos = convert_grid_id_to_pos(source, world, (-1, 1))
            landmark.state.p_vel = np.zeros(world.dim_p)

            s_numpy = np.array(source)
            t_numpy = np.array(target)
            landmark.price = np.sqrt((s_numpy - t_numpy) ** 2)  # Euler

            world.grids[source].update_landmarks([landmark])

        # collect landmarks
        world.landmarks = []
        for grid in world.grids.values():
            world.landmarks.extend(grid.landmarks.values())

    def update(self, world, time):
        """ Update agents' states and generate new agent or landmarks """

        self.time = time

        if self.time == 0:
            n_agents = number_generator(*self.agent_n_dist[self.time])
        else:
            n_agents = 0

        self.update_agent_domain(world, n_agents=n_agents)

        n_landmarks = number_generator(*self.landmark_n_dist[self.time])
        self.update_landmarks(world, n_landmarks)

    def reset_world(self, world):
        """Random properties for landmarks"""

        n_agents = number_generator(*self.agent_n_dist[self.time])
        n_landmarks = number_generator(*self.landmark_n_dist[self.time])

        self.update_landmarks(world, n_landmarks)
        self.update_agent_domain(world, n_agents)

        self.time = 0

    def reward(self, agent, world):
        """ Reward function """

        if agent.landmark is None:
            return 0.
        else:
            reward = agent.landmark.price if agent.landmark.target_grid_id == agent.grid_id else 0.
            agent.landmark = None
            agent.is_on_service = False

            return reward

    def observation(self, agent, world):
        """ Observation callback """

        grid_id = agent.grid_id

        return world.grids[grid_id].observation

    def match(self, landmarks, world):
        agent_ids, action_n = [], []

        for landmark in landmarks:

            # select agents
            source = landmark.grid_id
            target = landmark.target_grid_id

            for agent in world.grids[source].agents.values():
                if not agent.is_on_service:
                    landmark.done = True
                    agent.landmark = landmark
                    agent.is_on_service = True
                    agent_ids.append((source, agent.name))

                    action = np.where(np.array(target) - np.array(source) < 0, -1., 1.)

                    # extend * 2 then concatenate
                    action = np.concatenate([[0., 0.] if target[i] == source[i] else [action[i], 0.] for i in range(world.dim_p)])

                    action_n.append([0] + action.tolist())

        return agent_ids, action_n
