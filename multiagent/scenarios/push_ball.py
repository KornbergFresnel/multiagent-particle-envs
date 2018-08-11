import numpy as np

from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    """Push-Ball: A cooperative scenario with N agents and a big ball,
    all agents need to push a ball to a designated location by collision.
    """

    def __init__(self):
        super().__init__()
        self._dist_limit = 1e-4
    
    def make_world(self, **kwargs):
        """Generate a world instance

        :param kwargs: dict, keys: {'num_agents'}
        :return: World, the world instance
        """
        
        world = World()
        num_agents = kwargs['num_agents']
        num_landmarks = 1

        world.agents = [Agent() for _ in range(num_agents)]
        for i, agent in enumerate(world.agents):
            assert isinstance(agent, Agent)
            agent.name = 'push_ball_agent_%d' % i
            agent.collide = True
            agent.silent = True
            agent.color = np.array([0.75, 0.75, 0.75])
        
        # here the landmark is the ball needs to be pushed
        world.landmarks = [Landmark() for _ in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            assert isinstance(landmark, Landmark)
            landmark.name = 'push_ball_ball_%d' % i
            landmark.collide = True
            landmark.movable = True
        
        self.reset_world(world)
        return world
    
    def reset_world(self, world: World):
        """Random initialize the world"""

        for i, landmark in enumerate(world.landmarks):
            assert isinstance(landmark, Landmark)
            landmark.color = np.array([0.1, 0.1, 0.1])
            landmark.index = i
            landmark.state.p_pos = np.random.uniform(-1, 1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        
        # in this scenario, we set only one landmark as the target location
        # need to avoid the ball
        for i, agent in enumerate(world.agents):
            assert isinstance(agent, Agent)
            agent.state.p_pos = np.random.uniform(-1, 1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)

        world.target_location = np.random.uniform(-1, 1, world.dim_p)
        world.collaborative = True

    def reward(self, agent: Agent, world: World):
        """Get reward"""
        ball = world.landmarks[0]
        location = world.target_location
        # judge the distance of ball and target location
        distance = np.sqrt(np.sum((ball.state.p_pos - location) ** 2))
        return 1.0 if distance <= self._dist_limit else 0.0
    
    def observation(self, agent: Agent, world: World):
        """Get observation of agent"""

        ball = world.landmarks[0]

        relative_pos_to_ball = ball.state.p_pos - agent.state.p_pos
        obs_ball = np.concatenate([ball.color, ball.state.p_vel, ball.state.p_pos - world.target_location])
        obs_others = []

        for e in world.agents:
            if e is agent:
                continue
            obs_others.append(e.state.p_pos - agent.state.p_pos)

        obs_others = np.concatenate(obs_others)

        return np.concatenate([agent.state.p_vel, relative_pos_to_ball, obs_others, obs_ball])
