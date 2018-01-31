from railrl.torch.core import PyTorchModule


class ModelToImplicitModel(PyTorchModule):
    def __init__(self, model, bias=0):
        super().__init__()
        self.model = model
        self.bias = bias

    def forward(self, obs, action, next_obs):
        diff = next_obs - obs - self.model(obs, action)
        return -(diff**2).sum(dim=1, keepdim=True) + self.bias
