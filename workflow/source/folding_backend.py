from abc import ABC, abstractmethod


class FoldingBackend(ABC):
    @abstractmethod
    def predict(self, **kwargs):
        pass

    @abstractmethod
    def postprocess(self, **kwargs):
        pass
