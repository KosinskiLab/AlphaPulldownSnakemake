from typing import Dict

from alphafold_backend import AlphaFold


class FoldingBackendManager:
    def __init__(self):
        self._BACKEND_REGISTRY = {
            "alphafold": AlphaFold,
        }

    def change_backend(self, backend_name: str, **backend_kwargs: Dict) -> None:
        self._backend = self._BACKEND_REGISTRY[backend_name](**backend_kwargs)


backend = FoldingBackendManager()
