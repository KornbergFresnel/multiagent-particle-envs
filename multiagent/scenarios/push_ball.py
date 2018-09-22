import numpy as np

from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


BALL_INDEX = 0
TARGET_INDEX = 1
AGENT_INDEX = 2


class Scenario(BaseScenario):
    """Push-Ball: A cooperative scenario with N agents and a big ball,
    all agents need to push a ball to a designated location by collision.
    """

    def __init__(self):
        super().__init__()
        self._dist_limit = 1e-4
        self._agent_size = 1.
        self._agent_shaping = 0.08
        self._last_distance = np.inf

    def make_world(self, **kwargs):
        """Generate a world instance

        :param kwargs: dict, keys: {'num_agents'}
        :return: World, the world instance
        """

        world = World()
        num_agents = kwargs['num_agents']
        num_landmarks = 2

        world.agents = [Agent() for _ in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent_%d' % i
            agent.collide = True
            agent.silent = True
            agent.color = np.array([0.75, 0.75, 0.75])

        # here the landmark is the ball needs to be pushed
        world.landmarks = [Landmark() for _ in range(num_landmarks)]
        ball, target = world.landmarks

        ball.name = 'BALL'
        ball.collide = True
        ball.movable = True
        ball.color = np.array([229 / 255, 132 / 255, 129 / 255])
        ball.size *= 8.0
        ball.initial_mass *= 144

        self._agent_size = ball.size * self._agent_shaping

        target.name = 'TARGET'
        target.collide = False
        target.movable = False
        target.color = np.array([0.1, 0.1, 0.1])

        self.reset_world(world)
        return world

    def reset_world(self, world: World):
        """Random initialize the world"""
        cam_range = 2

        for i, landmark in enumerate(world.landmarks):
            size = landmark.size
            landmark.state.p_pos = np.random.uniform(
                -cam_range + size, cam_range - size, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
            landmark.index = BALL_INDEX

        world.landmarks[1].state.p_pos = np.array([0., 0.])
        world.landmarks[1].size = self._agent_size
        world.landmarks[1].index = TARGET_INDEX

        # in this scenario, we set only one landmark as the target location
        for i, agent in enumerate(world.agents):
            size = agent.size
            agent.state.p_pos = np.random.uniform(
                -cam_range + size, cam_range - size, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.size = self._agent_size
            agent.index = AGENT_INDEX

        world.target_location = np.array([0., 0.])
        world.collaborative = True
        self._last_distance = np.sqrt(np.sum(
            np.square(world.landmarks[0].state.p_pos - world.landmarks[1].state.p_pos)))

    def reward(self, agent: Agent, world: World):
        """Get reward"""
        reshaping = True
        alpha = 2.0

        ball = world.landmarks[0]
        location = world.target_location
        # judge the distance of ball and target location
        distance = np.sqrt(np.sum((ball.state.p_pos - location) ** 2))

        reward = self._last_distance - distance
        self._last_distance = distance

        if reshaping:
            reward *= alpha
        return reward

    @staticmethod
    def observation(agent: Agent, world: World, view_mode='vector'):
        """Get observation of agent"""

        if view_mode not in ['matrix', 'vector']:
            raise Exception('Unaccepted view mode: {}'.format(view_mode))
        if view_mode == 'matrix':
            return world.get_partial_view(agent, 3)
        else:
            # 2 cnn style
            ball, target = world.landmarks[0], world.landmarks[1]

            # format: {self, others, rel-ball, rel-target}

            # velocity channel
            vel_channel = []
            pos_channel = []

            vel_channel.append(agent.state.p_vel)
            pos_channel.append(agent.state.p_pos)

            for e in world.agents:
                if e is agent:
                    continue
                vel_channel.append(e.state.p_vel)
                pos_channel.append(e.state.p_pos)

            vel_channel.append(ball.state.p_vel)
            pos_channel.append(ball.state.p_pos)

            vel_channel.append(target.state.p_vel)
            pos_channel.append(target.state.p_pos)

            vel_channel = np.stack(vel_channel)
            pos_channel = np.stack(pos_channel)

            return np.stack([vel_channel, pos_channel], axis=2)
