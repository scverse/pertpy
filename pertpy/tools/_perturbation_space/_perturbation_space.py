from abc import ABC, abstractmethod


class PerturbationSpace(ABC):
    """Implements various ways of interacting with PerturbationSpaces.

    We differentiate between a cell space and a perturbation space.
    Visually speaking, in cell spaces single dota points in an embeddings summarize a cell,
    whereas in a perturbation space, data points summarize whole perturbations.
    """

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def add(self):
        raise NotImplementedError

    def subtract(self):
        raise NotImplementedError
