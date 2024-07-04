import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from src.dataset import get_dataloader
from src.model import CustomModel
from src.model_training_utils import save_checkpoint, load_checkpoint
import os

def train_model(model, dataloader, config):
    criterion = nn.PoissonNLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr_init'])
    writer = SummaryWriter()  # For logging

    model.to(config['device'])
    
    for epoch in range(config['max_iter']):
        model.train()
        running_loss = 0.0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(config['device']), targets.to(config['device'])
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(dataloader)
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        print(f'Epoch {epoch + 1}/{config["max_iter"]}, Loss: {epoch_loss:.4f}')

    writer.close()
    print('Training complete')

def evaluate_model(model, dataloader, config):
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(config['device']), targets.to(config['device'])
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == targets).sum().item()
            total_samples += targets.size(0)

    accuracy = total_correct / total_samples
    print(f'Accuracy: {accuracy:.4f}')

def main():
    data_dir = '/data/monkey/toliaslab/CSRF19_V4/sua_data/'
    neuronal_data_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith('.pickle')]

    dataset_config = {
        'dataset': 'CSRF19_V4',
        'neuronal_data_files': neuronal_data_files,
        'image_cache_path': '/data/monkey/toliaslab/CSRF19_V4/images/',
        'crop': [18, 18, 110, 110],  # Modify based on the crop shape used in the images
        'subsample': 1,
        'scale': 0.4,
        'time_bins_sum': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        'batch_size': 64,
        'seed': 1,
        'img_mean': 124.34,
        'img_std': 70.28,
        'stimulus_location': [-32, -1]
    }

    
    model_config = {
        'input_channels': 1,
        'model_name': 'resnet50_l2_eps0_1',
        'layer_name': 'layer3.0',
        'pretrained': True,
        'bias': False,
        'final_batchnorm': True,
        'final_nonlinearity': True,
        'momentum': 0.1,
        'fine_tune': False,
        'init_mu_range': 0.4,
        'init_sigma_range': 0.6,
        'readout_bias': True,
        'gamma_readout': 3.0,
        'gauss_type': 'isotropic',
        'elu_offset': -1,
        'data_info': None
    }

    trainer_config = {
        'stop_function': 'get_poisson_loss',
        'maximize': False,
        'avg_loss': False,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'max_iter': 100,
        'lr_init': 0.005,
        'lr_decay_steps': 4,
        'patience': 3,
        'verbose': True
    }

    dataloader = get_dataloader(dataset_config)
    model = CustomModel(model_config)

    # Load checkpoint if available
    optimizer = optim.Adam(model.parameters(), lr=trainer_config['lr_init'])
    start_epoch = load_checkpoint(model, optimizer)

    # Train the model
    train_model(model, dataloader, trainer_config)

    # Save the final model
    save_checkpoint(model, optimizer, trainer_config['max_iter'])

    # Evaluate the model
    evaluate_model(model, dataloader, trainer_config)

if __name__ == "__main__":
    main()