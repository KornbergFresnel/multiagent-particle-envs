import numpy as np

from multiagent.core import World, Agent, Landmark
from multiagent.scenarios import BaseScenario


class Scenario(BaseScenario):

    def make_world(self, **kwargs):
        world = World()

        world.dim_c = 2
        world.num_agents = kwargs["n_good_agents"]
        
        if kwargs["tranable_adverary"]:
            world.num_agents += kwargs["n_adversaries"]

        world.agents = [Agent() for i in range(world.num_agents)]

        n_good = kwargs["n_good_agents"]
        # n_adversaries = kwargs["n_adversaries"]

        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            
            agent.collide = False
            agent.silent = True
            agent.adversary = False if i < n_good else True
            agent.size = 0.15

        # add landmarks: one ball and two gate
        world.landmarks = [Landmark() for i in range(3)]
        for i, landmark in enumerate(world.landmarks):
            landmark.collide = True if i == 0 else False
            landmark.movable = True if i == 0 else False
            landmark.size = 0.08 if i == 0 else 0.30

        self.reset_world(world)

    def reset_world(self, world):
        # random properties for agents
        for i in range(world.num_agents):
            # TODO(ming): differentiate gate and other agents
            pass

        # ball and gate init
        # goal_for_good = [world.landmarks[0], world.landmarks[1]]
        # goal_for_adversary = [world.landmarks[0], world.landmarks[2]]

        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.15, 0.15, 0.15]) if i == 0 else np.array([0.35, 0.35, 0.35])            

        for agent in world.agents:
            agent.goal_ball = world.landmarks[0]
            agent.goal_gate = world.landmarks[2] if agent.adversary else world.landmarks[1]

    def benchmark(self, agent, world):
        pass

    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary] 

    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):
        callable_ins = self._adversary_reward if agent.adversary else self._agent_reward
        return callable_ins(agent, world)

    def done(self, agent, world):
        # TODO(ming): 1) if soccer out of world, done, 2) if soccer enter any gate, done
        ball = world.landmarks[0]

        return ball.out(world) or ball.at(world.landmarks[1:])

    def _adversary_reward(self, agent, world):
        raise NotImplementedError

    def _agent_reward(self, agent, world):
        raise NotImplementedError

    def observation(self, agent, world):
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        
        entity_color = []
        for entity in world.landmarks:
            entity_color.append(entity.color)

        other_pos = []
        for other in world.agents:
            if other is agent: continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)

        if not agent.adversary:
            return np.concatenate([agent.goal_ball.state.p_pos - agent.state.p_pos] + entity_pos + other_pos)
        else:
            return np.concatenate(entity_pos + other_pos)
        
            