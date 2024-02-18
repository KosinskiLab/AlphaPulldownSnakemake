from typing import Dict
from abc import ABC, abstractmethod

from alphapulldown.objects import MultimericObject
from alphapulldown.predict_structure import predict as af_predict, ModelsToRelax
from alphapulldown.utils import (
    get_run_alphafold,
    create_and_save_pae_plots,
    post_prediction_process,
)


class FoldingBackend(ABC):
    @abstractmethod
    def predict(self, **kwargs):
        pass

    @abstractmethod
    def postprocess(self, **kwargs):
        pass


class AlphaFold(FoldingBackend):
    def __init__(self, **kwargs):
        self._module = get_run_alphafold()

    @staticmethod
    def predict(**kwargs) -> None:
        if "models_to_relax" not in kwargs:
            kwargs["models_to_relax"] = ModelsToRelax
        return af_predict(**kwargs)

    @staticmethod
    def postprocess(
        multimer: MultimericObject,
        output_path: str,
        zip_pickles: bool = False,
        remove_pickles: bool = False,
        **kwargs: Dict
    ) -> None:
        create_and_save_pae_plots(multimer, output_path)
        post_prediction_process(
            output_path,
            zip_pickles=zip_pickles,
            remove_pickles=remove_pickles,
        )


class FoldingBackendManager:
    def __init__(self):
        self._BACKEND_REGISTRY = {
            "alphafold": AlphaFold,
        }

    def change_backend(self, backend_name: str, **backend_kwargs: Dict) -> None:
        self._backend = self._BACKEND_REGISTRY[backend_name](**backend_kwargs)


backend = FoldingBackend()
