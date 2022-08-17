import torch

class MF(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        model_configs=kwargs.pop('model_configs')
        self.num_users = model_configs['num_users']
        self.num_items = model_configs['num_items']
        self.embedding_size = model_configs['embedding_size']
        self.weight_decay = model_configs['weight_decay']

        # torch.random.manual_seed(0)
        self.user_embedding = torch.nn.Embedding(self.num_users, self.embedding_size)
        self.item_embedding = torch.nn.Embedding(self.num_items, self.embedding_size)

    def reset_parameters(self):
        self.user_embedding.reset_parameters()
        self.item_embedding.reset_parameters()

    def forward(self, x):
        user_ids = x[:, 0]
        item_ids = x[:, 1]
        x_users = self.user_embedding(user_ids)
        x_items = self.item_embedding(item_ids)
        rating = torch.sum(x_users*x_items, dim=1)
        return rating

    def get_loss_fn(self):
        return torch.nn.MSELoss()

    def get_optimizer(self, params=None, lr=1e-3):
        assert params != None
        return torch.optim.SGD(params, lr=lr)