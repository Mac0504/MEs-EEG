import os
import argparse
import yaml
from data.eeg_dataset import EEGDataset
from data.me_dataset import MEDataset
from data.utils import load_data_split, create_data_loaders
from models.eeg_feature_extractor import EEGFeatureExtractor
from models.me_feature_extractor import MEFeatureExtractor
from models.transformer_fusion import TransformerFusion
from train.train import train_epoch, validate_epoch, save_checkpoint
from train.loss import MultiModalLoss
from train.evaluate import evaluate_model
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import LeaveOneOut
from datetime import datetime

def load_config(config_path):
    """Load the configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Multimodal Emotion Recognition")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to the configuration file')
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'infer'], required=True, help='Mode to run the script in')
    parser.add_argument('--checkpoint', type=str, help='Path to the model checkpoint for evaluation or inference')
    return parser.parse_args()

def create_loso_split(dataset, subject_id):
    """
    Create training and validation sets for LOSO cross-validation.
    
    Args:
        dataset: Combined dataset with subject information
        subject_id: ID of subject to leave out for validation
    
    Returns:
        train_dataset: Dataset with all subjects except subject_id
        val_dataset: Dataset with only subject_id
    """
    train_indices = [i for i, sample in enumerate(dataset) if sample['subject_id'] != subject_id]
    val_indices = [i for i, sample in enumerate(dataset) if sample['subject_id'] == subject_id]
    return Subset(dataset, train_indices), Subset(dataset, val_indices)

def initialize_models_and_optimizers(config, device):
    """Initialize models, criterion, optimizer, and scheduler."""
    eeg_feature_extractor = EEGFeatureExtractor().to(device)
    me_feature_extractor = MEFeatureExtractor().to(device)
    fusion_model = TransformerFusion().to(device)
    
    criterion = MultiModalLoss(num_classes=config['model']['num_classes'], 
                             feature_dim=config['model']['feature_dim'])
    optimizer = optim.Adam([
        {'params': eeg_feature_extractor.parameters()},
        {'params': me_feature_extractor.parameters()},
        {'params': fusion_model.parameters()}
    ], lr=config['train']['learning_rate'])
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                        step_size=config['train']['step_size'], 
                                        gamma=config['train']['gamma'])
    
    return eeg_feature_extractor, me_feature_extractor, fusion_model, criterion, optimizer, scheduler

def train_with_loso(config, combined_dataset, device, log_dir_base):
    """Perform training with LOSO cross-validation."""
    subject_ids = sorted(set(sample['subject_id'] for sample in combined_dataset))
    loo = LeaveOneOut()
    batch_size = config['train']['batch_size']
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(loo.split(subject_ids)):
        print(f"\nStarting Fold {fold + 1}/{len(subject_ids)}")
        
        # Set up logging for this fold
        log_dir = os.path.join(log_dir_base, f"fold_{fold+1}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
        writer = SummaryWriter(log_dir)
        
        # Create datasets for this fold
        train_dataset, val_dataset = create_loso_split(combined_dataset, subject_ids[val_idx[0]])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Initialize models and training components
        eeg_feature_extractor, me_feature_extractor, fusion_model, criterion, optimizer, scheduler = \
            initialize_models_and_optimizers(config, device)
        
        best_val_loss = float('inf')
        for epoch in range(1, config['train']['num_epochs'] + 1):
            train_loss = train_epoch(fusion_model, train_loader, criterion, optimizer, device, epoch)
            val_loss = validate_epoch(fusion_model, val_loader, criterion, device, epoch)
            
            scheduler.step()
            
            writer.add_scalar('Train/Loss', train_loss, epoch)
            writer.add_scalar('Validation/Loss', val_loss, epoch)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(fusion_model, optimizer, epoch, os.path.join(log_dir, 'best_model.pth'))
        
        save_checkpoint(fusion_model, optimizer, config['train']['num_epochs'], 
                       os.path.join(log_dir, 'final_model.pth'))
        
        fold_results.append({'fold': fold + 1, 'best_val_loss': best_val_loss})
        writer.close()
    
    # Print summary
    print("\nLOSO Cross-Validation Results:")
    for result in fold_results:
        print(f"Fold {result['fold']}: Best Validation Loss = {result['best_val_loss']:.6f}")
    avg_val_loss = sum(r['best_val_loss'] for r in fold_results) / len(fold_results)
    print(f"Average Validation Loss across folds: {avg_val_loss:.6f}")
    
    return fold_results

def main():
    """Main function to run the multimodal emotion recognition pipeline with LOSO."""
    args = parse_args()
    config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load and combine datasets
    eeg_dataset = EEGDataset(config['data']['data/EEG'])
    me_dataset = MEDataset(config['data']['data/MEs'])
    # Note: This assumes datasets can be paired; adjust based on your actual data structure
    combined_dataset = list(zip(eeg_dataset, me_dataset))  # Temporary; implement proper pairing
    
    log_dir_base = os.path.join(config['logs']['log_dir'], 'logs')
    
    if args.mode == 'train':
        # Training mode with LOSO
        train_with_loso(config, combined_dataset, device, log_dir_base)
    
    elif args.mode == 'eval':
        # Evaluation mode (using a pre-trained model)
        if args.checkpoint is None:
            raise ValueError("Checkpoint path must be provided for evaluation mode.")
        
        # Prepare test data
        _, _, test_dataset = load_data_split(eeg_dataset, me_dataset)
        test_loader = DataLoader(test_dataset, batch_size=config['train']['batch_size'])
        
        # Initialize models
        eeg_feature_extractor, me_feature_extractor, fusion_model, criterion, optimizer, _ = \
            initialize_models_and_optimizers(config, device)
        
        # Load checkpoint
        checkpoint = torch.load(args.checkpoint)
        fusion_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        
        # Evaluate
        metrics = evaluate_model(fusion_model, test_loader)
        print(f"Evaluation Metrics at Epoch {epoch}:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    
    elif args.mode == 'infer':
        # Inference mode
        if args.checkpoint is None:
            raise ValueError("Checkpoint path must be provided for inference mode.")
        
        # Prepare test data
        _, _, test_dataset = load_data_split(eeg_dataset, me_dataset)
        test_loader = DataLoader(test_dataset, batch_size=config['train']['batch_size'])
        
        # Initialize models
        eeg_feature_extractor, me_feature_extractor, fusion_model, criterion, optimizer, _ = \
            initialize_models_and_optimizers(config, device)
        
        # Load checkpoint
        checkpoint = torch.load(args.checkpoint)
        fusion_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Perform inference
        fusion_model.eval()
        with torch.no_grad():
            for batch_idx, (eeg_data, me_data, labels) in enumerate(test_loader):
                eeg_data, me_data, labels = eeg_data.to(device), me_data.to(device), labels.to(device)
                eeg_features = eeg_feature_extractor(eeg_data)
                me_features = me_feature_extractor(me_data)
                outputs = fusion_model(eeg_features, me_features)
                _, preds = torch.max(outputs, 1)
                print(f"Batch {batch_idx + 1} Predictions: {preds.cpu().numpy()}")
                print(f"Batch {batch_idx + 1} Ground Truth: {labels.cpu().numpy()}")
                break  # Demo with first batch

if __name__ == "__main__":
    main()
