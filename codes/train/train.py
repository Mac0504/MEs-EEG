import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from models.eeg_feature_extractor import EEGFeatureExtractor
from models.me_feature_extractor import MEFeatureExtractor
from models.transformer_fusion import TransformerFusion
from data.eeg_dataset import EEGDataset
from data.me_dataset import MEDataset
from data.utils import load_data_split, create_data_loaders
from train.loss import MultiModalLoss
import yaml
from sklearn.model_selection import LeaveOneOut

# Load configuration
with open('../configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_epoch(model, data_loader, criterion, optimizer, device, epoch):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0.0

    for batch_idx, (eeg_data, me_data, labels) in enumerate(data_loader):
        eeg_data, me_data, labels = eeg_data.to(device), me_data.to(device), labels.to(device)
        optimizer.zero_grad()
        eeg_features = eeg_feature_extractor(eeg_data)
        me_features = me_feature_extractor(me_data)
        outputs = model(eeg_features, me_features)
        loss = criterion(outputs, eeg_features, me_features, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch_idx % config['train']['log_interval'] == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(eeg_data)}/{len(data_loader.dataset)} "
                  f"({100. * batch_idx / len(data_loader):.0f}%)]\tLoss: {loss.item():.6f}")
    return total_loss / len(data_loader)

def validate_epoch(model, data_loader, criterion, device, epoch):
    """Validate the model for one epoch."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for eeg_data, me_data, labels in data_loader:
            eeg_data, me_data, labels = eeg_data.to(device), me_data.to(device), labels.to(device)
            eeg_features = eeg_feature_extractor(eeg_data)
            me_features = me_feature_extractor(me_data)
            outputs = model(eeg_features, me_features)
            loss = criterion(outputs, eeg_features, me_features, labels)
            total_loss += loss.item()
    return total_loss / len(data_loader)

def save_checkpoint(model, optimizer, epoch, save_path):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, save_path)

def create_loso_split(dataset, subject_idx):
    """
    Create training and validation sets for LOSO cross-validation.
    
    Args:
        dataset: Combined EEG and ME dataset with subject information
        subject_idx: Index of subject to leave out for validation
    
    Returns:
        train_dataset: Dataset with all subjects except subject_idx
        val_dataset: Dataset with only subject_idx
    """
    train_indices = [i for i, sample in enumerate(dataset) if sample['subject_id'] != subject_idx]
    val_indices = [i for i, sample in enumerate(dataset) if sample['subject_id'] == subject_idx]
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    return train_dataset, val_dataset

def main():
    """Main function implementing LOSO cross-validation."""
    # Load datasets
    eeg_dataset = EEGDataset(config['data']['eeg_path'])
    me_dataset = MEDataset(config['data']['me_path'])
    
    # Assuming datasets have been combined with subject information
    combined_dataset = list(zip(eeg_dataset, me_dataset))  # This needs proper implementation based on your data structure
    
    # Get unique subject IDs (assuming this information is available in your dataset)
    subject_ids = sorted(set(sample['subject_id'] for sample in combined_dataset))
    
    # Set up LOSO cross-validation
    loo = LeaveOneOut()
    batch_size = config['train']['batch_size']
    
    # Store results for each fold
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(loo.split(subject_ids)):
        print(f"\nStarting Fold {fold + 1}/{len(subject_ids)}")
        
        # Create log directory for this fold
        log_dir = os.path.join(config['logs']['log_dir'], 
                             f"loso_fold_{fold+1}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
        writer = SummaryWriter(log_dir)
        
        # Create train and validation datasets for this fold
        train_dataset, val_dataset = create_loso_split(combined_dataset, subject_ids[val_idx[0]])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Initialize models for this fold
        eeg_feature_extractor = EEGFeatureExtractor().to(device)
        me_feature_extractor = MEFeatureExtractor().to(device)
        fusion_model = TransformerFusion().to(device)
        
        # Initialize optimizer
        optimizer = optim.Adam([
            {'params': eeg_feature_extractor.parameters()},
            {'params': me_feature_extractor.parameters()},
            {'params': fusion_model.parameters()}
        ], lr=config['train']['learning_rate'])
        
        # Initialize scheduler
        scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                            step_size=config['train']['step_size'], 
                                            gamma=config['train']['gamma'])
        
        # Initialize loss function
        criterion = MultiModalLoss(num_classes=config['model']['num_classes'], 
                                 feature_dim=config['model']['feature_dim'])
        
        # Training loop for this fold
        best_val_loss = float('inf')
        for epoch in range(1, config['train']['num_epochs'] + 1):
            train_loss = train_epoch(fusion_model, train_loader, criterion, optimizer, device, epoch)
            val_loss = validate_epoch(fusion_model, val_loader, criterion, device, epoch)
            
            scheduler.step()
            
            # Log metrics
            writer.add_scalar('Train/Loss', train_loss, epoch)
            writer.add_scalar('Validation/Loss', val_loss, epoch)
            
            # Save best model for this fold
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(fusion_model, optimizer, epoch, 
                              os.path.join(log_dir, 'best_model.pth'))
        
        fold_results.append({
            'fold': fold + 1,
            'best_val_loss': best_val_loss
        })
        
        # Save final model for this fold
        save_checkpoint(fusion_model, optimizer, config['train']['num_epochs'],
                       os.path.join(log_dir, 'final_model.pth'))
        
        writer.close()
    
    # Print summary of results
    print("\nLOSO Cross-Validation Results:")
    for result in fold_results:
        print(f"Fold {result['fold']}: Best Validation Loss = {result['best_val_loss']:.6f}")
    avg_val_loss = sum(r['best_val_loss'] for r in fold_results) / len(fold_results)
    print(f"Average Validation Loss across folds: {avg_val_loss:.6f}")

if __name__ == "__main__":
    main()
