"""Defines scenario upon which the world is built
"""


class BaseScenario(object):
    def __init__(self, *args, **kwargs):
        self.time = 0
        return super().__init__(*args, **kwargs)

    def make_world(self, **kwargs):
        # create elements of the world
        raise NotImplementedError

    def reset_world(self, world):
        # create initial conditions of the world
        raise NotImplementedError
