import torch
import torch.nn as nn
import torchvision.models as models

class TaskDrivenCore(nn.Module):
    """
    Core model based on ResNet-50 for feature extraction.
    
    Attributes:
        config (dict): Configuration dictionary for the model.
        resnet (nn.Module): ResNet-50 model.
        layer3 (nn.Module): Layer3 of the ResNet-50 model.
    """
    def __init__(self, config):
        """
        Initialize the core model with configuration parameters.

        Args:
            config (dict): Configuration dictionary for the model.
        """
        super(TaskDrivenCore, self).__init__()
        self.config = config
        self.resnet = models.resnet50(pretrained=config['pretrained'])
        self.layer3 = self.resnet.layer3[0]

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.layer3(x)
        return x
    
class Gaussian2dReadout(nn.Module):
    """
    Readout layer that applies a Gaussian filter over feature maps.
    
    Attributes:
        in_shape (tuple): Shape of the input.
        outdims (int): Number of output dimensions.
        bias (bool): Whether to include bias.
        init_mu_range (tuple): Range for initializing mu.
        init_sigma_range (tuple): Range for initializing sigma.
        gauss_type (str): Type of Gaussian ('isotropic' or 'anisotropic').
        mu (nn.Parameter): Mu parameter.
        sigma (nn.Parameter): Sigma parameter.
        weight (nn.Parameter): Weight parameter.
        bias (nn.Parameter): Bias parameter.
    """
    def __init__(self, in_shape, outdims, bias, init_mu_range, init_sigma_range, gauss_type):
        super(Gaussian2dReadout, self).__init__()
        self.outdims = outdims
        self.in_shape = in_shape
        self.gauss_type = gauss_type

        self.mu = nn.Parameter(torch.Tensor(outdims, 2))
        self.sigma = nn.Parameter(torch.Tensor(outdims, 2))
        self.weight = nn.Parameter(torch.Tensor(outdims, in_shape[0]))
        self.bias = nn.Parameter(torch.Tensor(outdims)) if bias else None

        self.init_mu_range = init_mu_range
        self.init_sigma_range = init_sigma_range

        self.initialize()

    def initialize(self):
        nn.init.uniform_(self.mu, -self.init_mu_range, self.init_mu_range)
        nn.init.uniform_(self.sigma, self.init_sigma_range[0], self.init_sigma_range[1])
        nn.init.xavier_normal_(self.weight)

        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, x):
        batch_size, _, width, height = x.size()
        device = x.device

        mu = torch.sigmoid(self.mu) * width
        sigma = torch.exp(self.sigma)

        grid_x, grid_y = torch.meshgrid(torch.arange(width), torch.arange(height), indexing='xy')
        grid = torch.stack([grid_x, grid_y], dim=-1).to(device)
        grid = grid.unsqueeze(0).unsqueeze(0).expand(batch_size, self.outdims, width, height, 2)

        mu = mu.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(grid)
        sigma = sigma.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(grid)

        norm = (grid - mu) / sigma
        norm = norm ** 2
        norm = norm.sum(dim=-1)

        if self.gauss_type == 'isotropic':
            gauss = torch.exp(-0.5 * norm)
        elif self.gauss_type == 'anisotropic':
            gauss = torch.exp(-0.5 * norm.prod(dim=-1))
        else:
            raise ValueError(f"Unknown gauss_type: {self.gauss_type}")

        gauss = gauss / gauss.sum(dim=[2, 3], keepdim=True)
        gauss = gauss.view(batch_size, self.outdims, -1)

        x = x.view(batch_size, -1, width * height)
        y = torch.bmm(self.weight, x)
        y = y + (gauss * y).sum(dim=-1)

        if self.bias is not None:
            y += self.bias

        return y

class CustomModel(nn.Module):
    def __init__(self, config):
        super(CustomModel, self).__init__()
        self.core = TaskDrivenCore(config)
        self.readout = Gaussian2dReadout(
            in_shape=(256, 28, 28),
            outdims=1,
            bias=config['readout_bias'],
            init_mu_range=(config['init_mu_range'], config['init_sigma_range']),
            init_sigma_range=(config['init_mu_range'], config['init_sigma_range']),
            gauss_type=config['gauss_type']
        )

    def forward(self, x):
        x = self.core(x)
        x = self.readout(x)
        return x
