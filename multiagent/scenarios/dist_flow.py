import os.path as osp
import numpy as np

from copy import copy

from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


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

    def update_agent(remove_ids=None, add_agents=None):
        removed, migrates = [], []

        if remove_ids is not None:
            for key in remove_ids:
                agent = self._agents.pop(key)

                assert not agent.is_on_service, "Agent cannot remove since it is on serving!"
                removed.add(key)

        if add_ids is not None:
            for agent in add_agents:
                # TODO(ming): update agent info
                agent.callback("update_info", (self,))
                self._agents[agent.name] = agent


class Scenario(BaseScenario):
    def make_world(self, **kwargs):
        """Generate a world instance

        :param kwargs: dict, includes {'height', 'width', 'n_agents', 'n_landmarks'}
        :return: world, World, the world instance
        """

        world = World()
        world.grids = dict()

        for i in range(kwargs['width']):
            for j in range(kwargs['height']):
                world.grids[(i, j)] = Node(i, j, i * kwargs['width'] + j)

        n_agents = kwargs.get('n_agents', 0)
        n_landmarks = kwargs.get('n_landmarks', 0)

        # add agents
        self.reset_world(world, n_agents, n_landmarks)
        self.time_flag = 0

        return world

    def update_agent_domain(self, world, n_agents=0):
        # TODO(ming): need add callback function for each new agents

        if n_agents is not None:
            world.agents = [Agent() for _ in range(n_agents)]

            for i, agent in enumerate(world.agents):
                agent.name = 'agent_%d_%d' % (i, self.time_flag)
                agent.collide = False
                agent.silent = True
                agent.landmark = None
                agent.bind_callback("update_info", update_info)

        for agent in world.agents:
            # check domain
            if getattr(agent, "grid_id", None) is not None:
                agent.grid_id = check_grid(agent.state.p_pos)
            else:
                agent.grid_id = select_grid("random", world.grids)

            if agent.landmark is not None and agent.landmark.grid_id == agent.grid_id:
                # is agent is on its target domain, stop it
                agent.state.p_vel -= agent.state.p_vel

            # random stop
            if not agent.is_on_service:
                # TODO(ming): random stop

        for node in world.grids.values():  # since agent changed their domain, grid needs to remove these agents
            node.update_agent()

    def update_landmarks(self, world, n_landmarks):
        world.landmarks = [Landmark() for _ in range(n_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.price = 0

            # TODO(ming): dispatching this landmark to a certain grid
            landmark.grid_id = None

    def reset_world(self, world, n_agents=None, n_landmarks=None):
        """Random properties for landmarks"""

        self.update_landmarks(world, n_landmarks)
        self.update_agent_domain(world, n_agents)

        self.time_flag = 0

    def reward(self, agent, world):
        if self.agent.landmark is None:
            return 0
        else:
            return agent.landmark.price if agent.landmark.grid_id == agent.grid_id else 0

    def observation(self, agent, world):
        grid_id = agent.grid_id

        return self.grids[grid_id].observation

