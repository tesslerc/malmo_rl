from abc import ABC, abstractmethod


class Policy(ABC):
    """A policy abstract base class.
    Defines the methods which are required to be implemented by all policies deriving from this class.
    """
    def __init__(self, params):
        self.params = params

    @abstractmethod
    def get_action(self, reward, terminal, state, is_train):
        pass
