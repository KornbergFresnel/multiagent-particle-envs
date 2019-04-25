import os.path as osp
import numpy as np

from copy import copy

from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


def number_generator(mean, var=None, dist_type="gaussian"):
    # TODO(ming): Gaussian distribution

    if dist_type == "gaussian":
        n = np.random.normal(mean, var)
    else:
        raise NotImplementedError

    return n


def random_selector(dataset, p=None, size=None):
    # TODO(ming): Gaussian distribution
    return NotImplementedError


def check_grid(pos, unit):
    x, y = pos[0] // unit, pos[1] // unit
    return x, y


def update_info(node):
    raise NotImplementedError


class Node(object):
    def __init__(self, x, y, node_id):
        self._reabable_node_id = (x, y)
        self._node_id = node_id

        self._agents = dict()
        self._landmarks = dict()

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
                agent.callback("update_info", (self,))
                self._agents[agent.name] = agent

    def update_landmarks(self, landmarks=None):
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
        world.grids = dict()
        world.height, world.width, world.unit = kwargs['height'], kwargs['width'], kwargs['unit']

        for i in range(kwargs['width']):
            for j in range(kwargs['height']):
                world.grids[(i, j)] = Node(i, j, i * kwargs['width'] + j)

        n_agents = kwargs.get('n_agents', 0)
        n_landmarks = kwargs.get('n_landmarks', 0)

        self.landmark_s_dist = [None for _ in range(kwargs['max_steps'])]
        self.landmark_t_dist = [None for _ in range(kwargs['max_steps'])]
        self.agent_n_dist = [None for _ in range(kwargs['max_steps'])]
        self.landmark_n_dist = [None for _ in range(kwargs['max_steps'])]

        # add agents
        self.reset_world(world, n_agents, n_landmarks)
        self.time = 0

        return world

    def update_agent_domain(self, world, n_agents=0):
        # TODO(ming): need add callback function for each new agents, then collect ava_agents

        if n_agents is not None:
            world.agents = [Agent() for _ in range(n_agents)]

            for i, agent in enumerate(world.agents):
                agent.name = 'agent_%d_%d' % (i, self.time)
                agent.collide = False
                agent.silent = True
                agent.landmark = None
                agent.bind_callback("update_info", update_info)

        for agent in world.agents:
            # check domain
            if getattr(agent, "grid_id", None) is not None:
                agent.grid_id = check_grid(agent.state.p_pos, world.unit)
            else:
                agent.grid_id = random_selector(list(world.grids.keys()), p=None, size=1)

            if agent.landmark is not None and agent.landmark.target_grid_id == agent.grid_id:
                # is agent is on its target domain, stop it
                agent.state.p_vel -= agent.state.p_vel
            else:
                # random_stop() or random_start()
                if not agent.is_on_service:
                    # TODO(ming): random stop
                    raise NotImplementedError
                else:
                    # random_migrate() avoid
                    pass

        for node in world.grids.values():  # since agent changed their domain, grid needs to remove these agents
            node.update_agent()

    def update_landmarks(self, world, n_landmarks=0):
        landmarks = [Landmark() for _ in range(n_landmarks)]

        s_grids = random_selector(list(world.grids.keys()), p=self.landmark_s_dist[self.time], size=n_landmarks)
        t_grids = random_selector(list(world.grids.keys()), p=self.landmark_t_dist[self.time], size=n_landmarks)

        for i, landmark, source, target in enumerate(landmarks, s_grids, t_grids):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.grid_id = source
            landmark.target_grid_id = target

            s_numpy = np.array(source)
            t_numpy = np.array(target)
            landmark.price = np.sqrt((s_numpy - t_numpy) ** 2)  # Euler
            world.grids[source].update_landmarks([landmark])

        # collect landmarks
        world.landmarks = []
        for grid in world.grids:
            world.landmarks.extend(grid.landmark.values())

    def update(self, world, time):
        n_agents = number_generator(self.agent_n_dist[self.time])
        self.update_agent_domain(world, n_agents=n_agents)

        n_landmarks = number_generator(self.landmark_n_dist[self.time])
        self.update_landmarks(world, n_landmarks)
        self.time = time

    def reset_world(self, world, n_agents=None, n_landmarks=None):
        """Random properties for landmarks"""

        self.update_landmarks(world, n_landmarks)
        self.update_agent_domain(world, n_agents)

        self.time = 0

    def reward(self, agent, world):
        if agent.landmark is None:
            return 0
        else:
            return agent.landmark.price if agent.landmark.target_grid_id == agent.grid_id else 0

    def observation(self, agent, world):
        grid_id = agent.grid_id

        return world.grids[grid_id].observation
