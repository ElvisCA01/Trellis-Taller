from typing import *
import torch
import torch.nn as nn
from .. import models


class Pipeline:
    """
    A base class for pipelines.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
    ):
        if models is None:
            return
        self.models = models
        for model in self.models.values():
            model.eval()

    @staticmethod
    def from_pretrained(path: str) -> "Pipeline":
        """
        Load a pretrained model.
        """
        import os
        import json
        is_local = os.path.exists(f"{path}/pipeline.json")

        if is_local:
            config_file = f"{path}/pipeline.json"
        else:
            from huggingface_hub import hf_hub_download
            config_file = hf_hub_download(path, "pipeline.json")

        with open(config_file, 'r') as f:
            args = json.load(f)['args']

        _models = {
            k: models.from_pretrained(f"{path}/{v}")
            for k, v in args['models'].items()
        }

        new_pipeline = Pipeline(_models)
        new_pipeline._pretrained_args = args
        return new_pipeline

    @property
    def device(self) -> torch.device:
        # Jan 2025 memory optimizations: we'll move different models between CPU and GPU.
        # Our models are kept on CPU, but a model that is active is always loaded into GPU.
        # return 'cuda' if there is at least 1 model on CUDA. 
        # Only return 'cpu' if everything is on CPU:
        for model in self.models.values():
            if hasattr(model, 'device') and model.device.type == 'cuda':
                return torch.device('cuda')
            if hasattr(model, 'parameters'):
                try:
                    if next(model.parameters()).device.type == 'cuda':
                        return torch.device('cuda')
                except StopIteration:
                    continue # Handle models with no parameters.
        return torch.device('cpu')# If we get here, no models were on cuda.

    def to(self, device: torch.device) -> None:
        for model in self.models.values():
            model.to(device)

    def cuda(self) -> None:
         self.to(torch.device("cuda"))

    def cpu(self) -> None:
        self.to(torch.device("cpu"))
