import numpy as np

from collections import namedtuple


# Area support rectangle and circle
# for circle: type='circle', pos=list(), radius=float
# for rectangle: type='rec', pos=list(), raidus=[halfWidth, halfHeight]
Area = namedtuple("Area", "type, pos, radius")


class AreaAPI(object):

    def at(self, objection):
        raise NotImplementedError

    def out(self, objection):
        raise NotImplementedError

    def area(self):
        raise NotImplementedError


class EntityState(object):
    """Physical/external base state of all entities"""

    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None


class AgentState(EntityState):
    """State of agents (including communication and internal/mental state)"""

    def __init__(self):
        super(AgentState, self).__init__()
        # communication utterance
        self.c = None


class Action(object):
    """Action of the agent, include two types: physical action `self.u`
    and communication action `self.c`
    """

    def __init__(self):
        # physical action
        self.u = None
        # communication action
        self.c = None


class Entity(object):
    """Properties and state of physical world entity"""

    def __init__(self):
        # name
        self.name = ''
        # properties:
        self.size = 0.050
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # max speed and accelerate
        self.max_speed = None
        self.accel = None
        # state
        self.state = EntityState()
        # mass
        self.initial_mass = 1.0
        # identity status
        self.index = None

    @property
    def mass(self):
        return self.initial_mass


class Landmark(Entity, AreaAPI):
    """Properties of landmark entities"""

    def __init__(self):
        super(Landmark, self).__init__()

    def at(self, objection):
        assert hasattr(objection, "area")
        raise NotImplementedError

    def out(self, objection):
        assert hasattr(objection, "area")
        raise NotImplementedError

    def area(self):
        pos = self.state.p_pos
        return Area("circle", pos, self.size / 2.)


class Agent(Entity):
    """Properties of agent entities"""

    def __init__(self):
        super(Agent, self).__init__()
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1.0
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None
        # optional replay buffer
        self.replay_buffer = None

    def bind_callback(self, func_name, func_entity):
        assert getattr(func_name, None) is None, "Repeated function registion"
        setattr(self, func_name, func_entity)

    def callback(self, func_name, arg_list):
        assert getattr(self, func_name, None) is not None, "{} does not exist, pls check you've registed it".format(func_name)
        assert arg_list is None or isinstance(arg_list, tuple or list), "arg_list can be None or tuple or list"

        return getattr(self, func_name)(*arg_list)


class World(AreaAPI):
    """Multi-agent world"""

    def __init__(self):
        """List of agents and entities (can change at execution-time!)"""
        self.agents = []
        self.landmarks = []
        self.dim_c = 0  # communication channel dimensionality
        self.dim_p = 2  # position dimensionality
        self.dim_color = 3  # color dimensionality
        self.dt = 0.025  # simulation time-step
        self.damping = 0.25  # physical damping
        self.contact_force = 1e+2  # contact response parameters
        self.contact_margin = 1e-3
        self.vel_decay = True
        self.width = 700
        self.height = 700

    @property
    def entities(self):
        """Return all entities in the world

        :return list, all entities
        """
        return self.agents + self.landmarks

    @property
    def policy_agents(self):
        """Return all agents controllable by external policies: the agent can make actions by themselves

        :return list, the list of available agents
        """
        return [agent for agent in self.agents if agent.action_callback is None]

    @property
    def scripted_agents(self):
        """Return all agents controlled by world scripts

        :return list, the list of world-scripts-controlled agents
        """
        return [agent for agent in self.agents if agent.action_callback is not None]

    def at(self, objection):
        assert hasattr(objection, "area")
        raise NotImplementedError

    def out(self, objection):
        assert hasattr(objection, "area")
        raise NotImplementedError

    def area(self):
        pos = self.state.p_pos
        return Area("circle", pos, self.size / 2.)

    def step(self):
        """Update state of the world"""
        # set actions for scripted agents
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
        # gather forces applied to entities
        p_force = [None] * len(self.entities)
        # apply agent physical controls
        p_force = self.apply_action_force(p_force)
        # apply environment forces
        p_force = self.apply_environment_force(p_force)
        # integrate physical state
        self.integrate_state(p_force)
        # update agent state
        for agent in self.agents:
            self.update_agent_state(agent)

    # gather agent action forces
    def apply_action_force(self, p_force: list):
        """ Gather agent action forces

        Parameters
        ----------
        p_force
            for forces storage

        Returns
        -------
        p_force
        """
        # set applied forces
        for i, agent in enumerate(self.agents):
            if agent.movable:
                noise = np.random.randn(*agent.action.u.shape) * agent.u_noise if agent.u_noise else 0.0

                if agent.action.u is None:
                    agent.action.u = np.zeros(self.dim_p)
                    agent.action.c = np.zeros(self.dim_c)

                p_force[i] = agent.action.u + noise
        return p_force

    def apply_environment_force(self, p_force: list):
        """Gather physical forces acting on entities, simple (but inefficient) collision response"""

        for a, entity_a in enumerate(self.entities):
            for b, entity_b in enumerate(self.entities):
                if b <= a:
                    continue

                [f_a, f_b] = self.get_collision_force(entity_a, entity_b)

                if f_a is not None:
                    if p_force[a] is None:
                        p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a]

                if f_b is not None:
                    if p_force[b] is None:
                        p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]
        return p_force

    def integrate_state(self, p_force: list):
        """ Integrate physical state

        Parameters
        ----------
        p_force
            for each entities
        """

        for i, entity in enumerate(self.entities):
            if not entity.movable:
                continue

            if self.vel_decay:
                entity.state.p_vel = entity.state.p_vel * (1 - self.damping)  # velocity decay

            if p_force[i] is not None:
                entity.state.p_vel += (p_force[i] / entity.mass) * self.dt

            if entity.max_speed is not None:  # speed rectify
                speed = np.sqrt(np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1]))
                if speed > entity.max_speed:
                    # entity.state.p_vel = entity.state.p_vel / np.sqrt(np.square(entity.state.p_vel[0]) +
                    #                                               np.square(entity.state.p_vel[1])) * entity.max_speed
                    entity.state.p_vel = entity.state.p_vel / speed * entity.max_speed

            # if 'agent' in entity.name and entity.landmark is not None:
            #     if entity.landmark.target_grid_id == entity.grid_id:
            #         return
            #     else:
            entity.state.p_pos += entity.state.p_vel * self.dt
            entity.state.p_pos = np.clip(entity.state.p_pos, -0.99, 0.99)

            if entity.state.p_pos[0] == -0.99 or entity.state.p_pos[0] == 0.99:
                entity.state.p_vel *= -1.

            if entity.state.p_pos[1] == -0.99 or entity.state.p_pos[1] == 0.99:
                entity.state.p_vel[1] *= -1.


    def update_agent_state(self, agent):
        """Set communication state (directly for now), if the entity has non-zero communication channel"""

        if self.dim_c > 0:
            if agent.silent:
                agent.state.c = np.zeros(self.dim_c)
            else:
                noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
                agent.state.c = agent.action.c + noise

    def get_collision_force(self, entity_a, entity_b):
        """Get collision forces for any contact between two entities, entity_a collide entity_b

        Parameters
        ----------
        entity_a: Entity, agent or landmark
        entity_b: Entity, agent or landmark

        Returns
        -------
        list, include two forces of entity_a and entity_b
        """

        if not entity_a.collide or not entity_b.collide:
            return [None, None]  # not a collider
        if entity_a is entity_b:
            return [None, None]  # don't collide against itself

        # compute actual distance between entities
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist = np.sqrt(np.sum(delta_pos ** 2))
        dist_min = entity_a.size + entity_b.size  # minimum allowable distance

        k = self.contact_margin  # SoftMax penetration
        penetration = np.logaddexp(0, -(dist - dist_min) / k) * k

        force = self.contact_force * delta_pos / dist * penetration
        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None

        return [force_a, force_b]

    def get_partial_view(self, agent: Agent, identity_kind: int):
        """ Get partial view of agent

        Parameters
        ----------
        agent
            the agent which get partial view

        identity_kind
            the sort of diffent entity

        Returns
        -------
        view_arr
            the matrix of partial view
        """

        # get the position of current agent
        center = agent.state.p_pos
        radius = 5
        unit_pixel = 0.1

        view_arr = np.zeros((radius * 2 + 1, radius * 2 + 1))
        center_idx = np.array([radius, radius])

        print('\n---new agent')

        for other in self.entities:
            assert isinstance(other, Entity)
            if other is agent:
                continue

            agent_pos = other.state.p_pos

            dis = agent_pos - center
            dis /= unit_pixel

            if np.all(np.abs(dis) <= radius):
                pos = center_idx + dis.astype(np.int32)
                view_arr[pos[0], pos[1]] = other.index
                print('entity type:', other.index)

        return view_arr


class DynamicWorld(World):
    """ Support dist_flow scenario
    """

    def __init__(self):
        super().__init__()
        del self.agents
        self.group = []

    @property
    def agents(self):
        """View of agents"""
        return list(self.group[0].values()) + list(self.group[1].values())

    @property
    def entities(self):
        return list(self.group[0].values()) + list(self.group[1].values()) + self.landmarks

    def apply_environment_force(self, p_force):
        """Gather physical forces acting on entities, and ignore collision force
        between agents which belong to the same group
        """

        for a, entity_a in enumerate(self.entities):
            for b, entity_b in enumerate(self.entities):
                if b <= a or entity_a.index == entity_b.index:
                    continue

                [f_a, f_b] = self.get_collision_force(entity_a, entity_b)

                if f_a is not None:
                    if p_force[a] is None:
                        p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a]

                if f_b is not None:
                    if p_force[b] is None:
                        p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]
        return p_force

    def get_collision_force(self, entity_a: Entity, entity_b: Entity):
        """Get collision forces and both update the hp
        """

        if not entity_a.collide or not entity_b.collide:
            return [None, None]

        if entity_a is entity_b:
            return [None, None]

        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos ** 2)))
        dist_min = entity_a.size + entity_b.size

        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min) / k) * k

        force = self.contact_force * delta_pos / dist * penetration

        force = self.contact_force * delta_pos / dist * penetration
        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None

        # focus on each of these two agents, if the focused agent move to the another agent,
        # and caused collide, we treat this movement as a attack, then the attacked agent will
        # loss some health point: 0.01
        pos_dir = delta_pos
        if np.all(pos_dir * entity_a.state.p_vel > 0):
            entity_b.hp -= entity_a.attack_value

        if np.all(-pos_dir * entity_b.state.p_vel > 0):
            entity_a.hp -= entity_b.attack_value

        return [force_a, force_b]
