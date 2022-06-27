import torch

class NCF(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        model_configs=kwargs.pop('model_configs')
        self.num_users = model_configs['num_users']
        self.num_items = model_configs['num_items']
        self.embedding_size = model_configs['embedding_size']
        self.weight_decay = model_configs['weight_decay']

    def initialize_parameters(self, seed=0):
        torch.random.seed(seed)
        pass

    def forward(self, idx):
        pass

    def get_loss_fn(self):
        return torch.nn.MSELoss()

    def get_optimizer(self, lr=1e-3):
        return torch.optim.SGD(lr)