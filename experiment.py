from model.model_factory import ModelFactory
from data_loader.data_loader import create_dataloader
from trainer import create_trainer
import itertools
import datetime
import json
import os
from utils import load_json, save_json
import torch

def get_curr_time():
    return datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

class Experiment:

    def __init__(self, 
                 train_packet, val_packet, test_packet,
                 device, num_workers, training_config=None,
                 rank=0):
        self.model_factory = ModelFactory()
        self.train_packet = train_packet
        self.val_packet = val_packet
        self.test_packet = test_packet

        self.device = device
        self.num_workers = num_workers
        self.rank = rank
        self.training_config = training_config if training_config else {
                'batch_size': 16,
                'lr': 1e-4,
                'num_workers': self.num_workers,
                'save_frequency': 10,
                'val_frequency': 4,
                'sequential': True,
                'seq_val_frequency': 110,
                'epochs': 100,
                # Early stopping configuration
                'early_stopping_enabled': True,
                'early_stopping_patience': 10,
                'early_stopping_min_delta': 0.001,
                'early_stopping_metric': 'loss',
                'early_stopping_mode': 'min',
            }

    def create_experiment_name(self, experiment_params):
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        experiment_name = experiment_params["model_name"]

        name_params = []
        for v in experiment_params.values():
            if type(v) != list:
                name_params.append(str(v))
            else:
                tmp = [str(s) for s in v]
                name_params.append("_".join(tmp))
        experiment_name = "_".join(name_params)
        experiment_name = f"{timestamp}_{experiment_name}"
        return experiment_name


    def load_model_and_training_type(self, experiment_params):
        state_dict = None
        if "state_dict" in experiment_params:
            state_dict = torch.load(experiment_params["state_dict"])["model_state_dict"]
            new_state_dict = {}
            for k, v in state_dict.items():
                new_state_dict[k.replace("module._orig_mod.", "")] = v
            state_dict = new_state_dict
        return self.model_factory.create_model(
            experiment_params["model_name"], 
            experiment_params, self.device, state_dict)


    def run_experiment_with_params(self, experiment_params, name=""):
        if not name:
            experiment_name = self.create_experiment_name(experiment_params)
        else:
            experiment_name = f"{name}_{get_curr_time()}"
        
        training_config = self.training_config
        training_config["experiment_name"] = experiment_name
        if "train_config" in experiment_params:
            for k, v in experiment_params["train_config"].items():
                training_config[k] = v

        if not os.path.exists(f"logs/{experiment_name}") and self.rank == 0:
            os.makedirs(f"logs/{experiment_name}")
            save_json(experiment_params, f"logs/{experiment_name}/params.json")
            save_json(training_config, f"logs/{experiment_name}/training_config.json")
        
        model, model_type = self.load_model_and_training_type(experiment_params)
        if training_config.get("compile", True):
            model = torch.compile(model, dynamic=False)
        if training_config.get("enable_parallel", False):
            if training_config.get("world_size", 1) > 1:
                # Distributed training with DDP
                rank = training_config.get("rank", 0)
                gpu_ids = training_config.get("gpu_ids", [0])
                if rank >= len(gpu_ids):
                    raise ValueError(f"Rank {rank} is out of range for available GPUs {gpu_ids}")
                # Always use cuda:0 since we've set CUDA_VISIBLE_DEVICES
                device_id = 0
                print(f"Process rank {rank} using GPU {gpu_ids[rank]}")
                model = torch.nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=[device_id],
                    output_device=device_id,
                    find_unused_parameters=True  # Enable unused parameter detection
                )
            else:
                # Single GPU training
                model = model.to(self.device)
        else:
            # CPU training
            model = model.to(self.device)
        trainer = create_trainer(self.train_packet, self.val_packet, self.test_packet, 
                                model, training_config, self.device,
                                model_type=model_type,
                                rank=self.rank)
        model = trainer.train(training_config["epochs"])
        results = trainer.evaluate(model)
        if self.rank == 0:
            print("Test Results:")
            print(results)
            save_json(results, f"logs/{experiment_name}/results.json")
            if training_config.get("sequential", False):
                print("Evaluating Sequential Model")
                seq_results = trainer.sequential_evaluate(model)
                print("Sequential Test Results:")
                print(seq_results)
                save_json(seq_results, f"logs/{experiment_name}/seq_results.json")


    def run_experiment(self, experiment_params):
        

        for k, v in experiment_params.items():
            if type(v) != list:
                experiment_params[k] = [v]
        
        combinations = list(itertools.product(*experiment_params.values()))
        for combination in combinations:
            param_dict = dict(zip(experiment_params.keys(), combination))
            self.run_experiment_with_params(param_dict)


    def run_experiment_with_config(self, config_path, config_name=""):
        """
        Run experiment with config file.
        If config_name is provided, run experiment with the given config name.
        Otherwise, run experiment with all configs in the config file.
        TODO: add support for multiple configs for a single config. For example
        "num_classes": [1, 5]
        should run the experiment twice, once with num_classes = 1 and once with num_classes = 5.
        """
        if type(config_path) == str:
            experiment_params = load_json(config_path)
        else:
            experiment_params = config_path
        if config_name:
            self.run_experiment_with_params(experiment_params[config_name], config_name)
            return
        for k, v in experiment_params.items():
            self.run_experiment_with_params(v, k)
