import os
import torch
from torch.utils.data import DataLoader
import models
from dataset import vessel_dataset
from tester import Tester
from utils import losses
from utils.helpers import get_instance
import copy
import ruamel.yaml
from bunch import Bunch

class FederatedServer:
    def __init__(self, CFG):
        self.global_model = get_instance(models, 'model', CFG)
        self.loss = get_instance(losses, 'loss', CFG)
        self.CFG = CFG
        self.clients = []  # This will store tuples of (client model weights, client optimizer state)

    def register_client_from_file(self, filepath):
        print(f"Attempting to load client model from {filepath}")
        client_checkpoint = torch.load(filepath)
        client_weights = client_checkpoint['state_dict']
        client_weights = {k.replace('module.', ''): v for k, v in client_weights.items()}
        client_optimizer_state = client_checkpoint.get('optimizer', None)
        self.register_client(client_weights, client_optimizer_state)
        print(f"Client model loaded and registered from {filepath}")

    def register_client(self, client_weights, client_optimizer_state=None):
        self.clients.append((client_weights, client_optimizer_state))  # Store as a tuple

    def aggregate_optimizer_state(self):
        if not self.clients:
            print("No clients registered. Cannot aggregate optimizer state.")
            return {}

        # Assuming all clients have the same keys in their optimizer state
        aggregated_state = {}
        for state_key in self.clients[0][1].keys():
            state_sum = {k: torch.zeros_like(v) for k, v in self.clients[0][1][state_key].items()}
            for client_weights, client_optimizer_state in self.clients:
                for k, v in client_optimizer_state[state_key].items():
                    state_sum[k] += v
            for k in state_sum.keys():
                state_sum[k] /= len(self.clients)
            aggregated_state[state_key] = state_sum
        return aggregated_state

    def aggregate_weights(self):
        total_clients = len(self.clients)
        for name, param in self.global_model.named_parameters():
            client_weights = [client[0][name] for client in self.clients if name in client[0]]
            if len(client_weights) == total_clients:
                client_sum = sum(client_weights)
                param.data = client_sum / total_clients
            else:
                print(f"Key {name} not found in all clients")
        self.clients = []  # Clear clients after aggregation

    def save_aggregated_model(self, save_path, use_data_parallel=False):
        model_state_dict = self.global_model.state_dict()
        if use_data_parallel:
            model_state_dict = {'module.' + k: v for k, v in model_state_dict.items()}

        aggregated_optimizer_state = self.aggregate_optimizer_state()
        checkpoint = {
            'arch': type(self.global_model).__name__,
            'epoch': 1,  # Placeholder, adjust as needed
            'state_dict': model_state_dict,
            'optimizer': aggregated_optimizer_state,
            'config': self.CFG
        }
        torch.save(checkpoint, save_path)
        print(f"Aggregated model checkpoint saved to {save_path}")

    def evaluate(self, test_loader):
        tester = Tester(self.global_model, self.loss, self.CFG, {'state_dict': self.global_model.state_dict()}, test_loader, None, show=False)
        tester.test()

# Example usage:
with open("config.yaml", encoding="utf-8") as file:
    yaml_instance = ruamel.yaml.YAML(typ='safe', pure=True)
    CFG = Bunch(yaml_instance.load(file))

server = FederatedServer(CFG)

client_files = ["saved/FR_UNet/231203142713/checkpoint-epoch1.pth", "saved/FR_UNet/231203142713/checkpoint-epoch1.pth"]
for filepath in client_files:
    server.register_client_from_file(filepath)

server.aggregate_weights()
server.save_aggregated_model("aggregated_model.pth", use_data_parallel=True)

# Optionally evaluate the global model performance
# ...
