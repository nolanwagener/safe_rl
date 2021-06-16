from enum import Enum


class Intervener:
    """
    An intervention mechanism.
    """

    class MODE(Enum):
        SAFE_ACTION = 'SAFE_ACTION'
        TELEPORT = 'TELEPORT'
        TERMINATE = 'TERMINATE'

    def __init__(self, mode=MODE.TERMINATE, **kwargs):
        self.mode = mode

    def reset(self, **kwargs):
        pass

    def set_state(self, env_state):
        pass

    def should_intervene(self, action):
        return False

    def safe_action(self):
        raise NotImplementedError
