
import torch
from model.autoregressive_transformer import AutoRegressiveTransformer
from enum import Enum
class ModelType(Enum):
    MULTI_CLASSES = "multi_classes"

class ModelFactory:

    def load_model(self, model_name, model_path, device):
        ckpt = torch.load(model_path)
        state_dict = ckpt["model_state_dict"]
        return AutoRegressiveTransformer.load_state_dict(state_dict).to(device), ModelType.MULTI_CLASSES

    def create_model(self, 
                    model_name, 
                    model_config, 
                    device,
                    state_dict=None):

        model_type = None
        model = AutoRegressiveTransformer(**model_config).to(device)
        model_type = ModelType.MULTI_CLASSES

        if state_dict:
            print("Loading state dict")
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("module._orig_mod."):
                    new_state_dict[k.replace("module._orig_mod.", "")] = v
                elif k.startswith("module."):
                    new_state_dict[k.replace("module.", "")] = v
                else:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict, strict=False)
        return model, model_type

