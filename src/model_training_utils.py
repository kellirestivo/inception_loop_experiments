import os
import torch

def save_checkpoint(model, optimizer, epoch, path='checkpoint.pth'):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    print(f'Checkpoint saved at epoch {epoch}')

def load_checkpoint(model, optimizer, path='checkpoint.pth'):
    if os.path.isfile(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f'Checkpoint loaded from epoch {start_epoch - 1}')
        return start_epoch
    else:
        print(f'No checkpoint found at {path}')
        return 0