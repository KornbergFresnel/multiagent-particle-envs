"""Defines scenario upon which the world is built
"""


class BaseScenario(object):
    def make_world(self, **kwargs):
        # create elements of the world
        raise NotImplementedError()

    def reset_world(self, world):
        # create initial conditions of the world
        raise NotImplementedError()
