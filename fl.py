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
        self.clients = []  # This will store client model weights

    def register_client_from_file(self, filepath):
        client_weights = torch.load(filepath)['state_dict']
        # Remove 'module.' prefix if present
        client_weights = {k.replace('module.', ''): v for k, v in client_weights.items()}
        self.register_client(client_weights)

    def register_client(self, client_weights):
        self.clients.append(client_weights)

    def aggregate_weights(self):
        """
        Aggregate weights from clients and update the global model.
        Handles missing keys by skipping them.
        """
        total_clients = len(self.clients)
        
        for name, param in self.global_model.named_parameters():
            client_weights = [client[name] for client in self.clients if name in client]

            if len(client_weights) == total_clients:
                client_sum = sum(client_weights)
                param.data = client_sum / total_clients
            else:
                print(f"Key {name} not found in all clients")

        self.clients = []

    def save_aggregated_model(self, save_path, use_data_parallel=False, epoch=None, optimizer_state=None, arch=None):
        """
        Save the aggregated model to a .pth file.
        Include model state_dict, epoch, optimizer_state, arch, and CFG (config).
        If use_data_parallel is True, add 'module.' prefix.
        """
        model_state_dict = self.global_model.state_dict()
        if use_data_parallel:
            # Add 'module.' prefix
            model_state_dict = {'module.' + k: v for k, v in model_state_dict.items()}

        checkpoint = {
            'arch': arch if arch else type(self.global_model).__name__,
            'epoch': epoch if epoch is not None else -1,  # Use -1 or another placeholder if epoch is not applicable
            'state_dict': model_state_dict,
            'optimizer': optimizer_state if optimizer_state else {},  # Empty dict or some default state if optimizer state is not applicable
            'config': self.CFG  # Save CFG as part of the checkpoint
        }
        torch.save(checkpoint, save_path)
        print(f"Aggregated model checkpoint saved to {save_path}")

    def evaluate(self, test_loader):
        tester = Tester(self.global_model, self.loss, self.CFG,
                        {'state_dict': self.global_model.state_dict()},
                        test_loader, None, show=False)
        tester.test()

# Example usage:
with open("config.yaml", encoding="utf-8") as file:
    yaml_instance = ruamel.yaml.YAML(typ='safe', pure=True)
    CFG = Bunch(yaml_instance.load(file))

server = FederatedServer(CFG)

client_files = ["saved/FR_UNet/dir/checkpoint-epoch1.pth", "saved/FR_UNet/dir/checkpoint-epoch1.pth"]
for filepath in client_files:
    server.register_client_from_file(filepath)

server.aggregate_weights()
# Set use_data_parallel to True if the model is intended to be used with DataParallel
server.save_aggregated_model("aggregated_model.pth", use_data_parallel=True)

# Optionally evaluate the global model performance
# ...
