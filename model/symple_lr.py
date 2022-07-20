import torch

class lr(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        model_configs=kwargs.pop('model_configs')
        self.linear = torch.nn.Linear(model_configs['input_elem'], 1)
        
    def reset_parameters(self):
        self.linear.reset_parameters()

    def forward(self, x):
        rating = self.linear(x.float())
        return rating.reshape(-1)

    def get_loss_fn(self):
        # return torch.nn.MSELoss()
        def loss_fn(pred, real):
            # print(pred)
            # print(real)
            # print(torch.square(pred-real))
            # print(torch.mean(torch.square(pred-real)))
            return torch.mean(torch.square(pred-real))
        return loss_fn

    def get_optimizer(self, params=None, lr=1e-3):
        assert params != None
        return torch.optim.SGD(params, lr=lr)