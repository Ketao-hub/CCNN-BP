import os
from RadarData import create_data_loaders
from RadarTransformer import RadarModel
from torch.multiprocessing import freeze_support
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from shape_and_mse_loss import ShapeBasedLoss, CombinedLoss


def compute_l2_loss(model, lambda_l2):
    
    device = next(model.parameters()).device
    l2_loss = torch.tensor(0., requires_grad=True).to(device)
    for param in model.parameters():
        l2_loss = l2_loss + torch.norm(param, p=2) ** 2  
    return lambda_l2 * l2_loss


def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        print("CUDA version:", torch.version.cuda)
        print("cuDNN version:", torch.backends.cudnn.version())
        
  
    max_epochs = 600
    # MiniBatchSize = 128
    MiniBatchSize = 100
    InitialLearnRate = 1e-3
    lambda_l2 = 1e-5


    save_dir = 'checkpoints'
    save_interval = 5 
    test_interval = 3  
    os.makedirs(save_dir, exist_ok=True)
    

    dataset_folder = "C:/matlab_data_all"
   
    i_data_dir_train = os.path.join(dataset_folder, "trainval_2", "radar", "i_data")
    q_data_dir_train = os.path.join(dataset_folder, "trainval_2", "radar", "q_data")
    tfm_bp_dir_train = os.path.join(dataset_folder, "trainval_2", "tfm_bp_no_fit")

    i_data_dir_test = os.path.join(dataset_folder, "test_2", "radar", "i_data")
    q_data_dir_test = os.path.join(dataset_folder, "test_2", "radar", "q_data")
    tfm_bp_dir_test = os.path.join(dataset_folder, "test_2", "tfm_bp_no_fit")

    train_loader = create_data_loaders(i_data_dir_train, q_data_dir_train, tfm_bp_dir_train, batch_size=MiniBatchSize)
    test_loader = create_data_loaders(i_data_dir_test, q_data_dir_test, tfm_bp_dir_test, batch_size=MiniBatchSize)

    model = RadarModel()
    model = model.to(device)  # 确保模型移到正确的设备上
    
    model = model.float()
    
    optimizer = optim.Adam(model.parameters(), lr=InitialLearnRate)
    criterion = CombinedLoss(shape_weight=0.5, mse_weight=0.5)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',          
        factor=0.5,          
        patience=20,         
        verbose=True,       
        min_lr=1e-6        
    )


    start_epoch = 0
    latest_model = None
    
    if os.path.exists(save_dir):
        checkpoint_files = [f for f in os.listdir(save_dir) if f.startswith('model_epoch_') and f.endswith('.pth')]
        if checkpoint_files:
            latest_model = max(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))

            checkpoint = torch.load(os.path.join(save_dir, latest_model))
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:  
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            

   
    
    log_dir = f'C:/TensorBoard/runs/radar_transformer_training'
    writer = SummaryWriter(log_dir, purge_step=None)  
    
    epochs = max_epochs
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        
        for i, (inputs, targets) in enumerate(train_loader):
           
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
          
            loss_shape_and_mse = criterion(outputs, targets)

            l2_loss = compute_l2_loss(model, lambda_l2)

            total_loss = loss_shape_and_mse + l2_loss
            
            if i % 10 == 9:  # 每10个批次打印一次
                print(f'Epoch [{epoch+1}/{epochs}], Batch [{i+1}/{len(train_loader)}]:')
                print(f'Shape and MSE Loss: {loss_shape_and_mse.item():.8f}')
                print(f'L2 Loss: {l2_loss.item():.8f}')
                print(f'Total Loss: {total_loss.item():.8f}')
                print('-' * 50)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()
            
            if i % 10 == 9:
                writer.add_scalar('Training/Batch Loss', 
                                total_loss.item(),
                                epoch * len(train_loader) + i)
        
        epoch_loss = running_loss/len(train_loader)
        writer.add_scalar('Training/Epoch Loss', 
                         epoch_loss,
                         epoch)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss}')
        
        if (epoch + 1) % test_interval == 0:
            model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    outputs = model(inputs)
                    test_loss += criterion(outputs, targets).item()
            
            avg_test_loss = test_loss / len(test_loader)
            scheduler.step(avg_test_loss)  
            
            writer.add_scalar('Testing/Epoch Loss',
                            avg_test_loss,
                            epoch)
        
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Training/Learning Rate',
                         current_lr,
                         epoch)
        
        if (epoch + 1) % save_interval == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),  
                'loss': epoch_loss,
            }
            torch.save(checkpoint, os.path.join(save_dir, f'model_epoch_{epoch+1}.pth'))
            

    writer.close()

if __name__ == '__main__':
    # freeze_support()
    main()