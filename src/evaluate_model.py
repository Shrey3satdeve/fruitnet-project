import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch_directml as dml

from dataset import FruitQualityDataset
from model import FruitNetMultiTask
from utils import accuracy

def evaluate_model(model_path, data_dir, batch_size=32):
    """Evaluate trained model on test set and calculate detailed accuracy metrics"""
    
    # Load checkpoint
    print("Loading model checkpoint...")
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"Warning: {e}. Loading with weights_only=False")
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Get mappings
    fruit2idx = checkpoint["fruit2idx"]
    quality2idx = checkpoint["quality2idx"]
    
    print(f"Loaded model with {len(fruit2idx)} fruit classes, {len(quality2idx)} quality classes")
    
    # Create dataset and dataloader
    dataset = FruitQualityDataset(data_dir, transform='inference')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Initialize model
    device = dml.device() if torch.cuda.is_available() else torch.device('cpu')
    model = FruitNetMultiTask(num_fruit_classes=len(fruit2idx), num_quality_classes=len(quality2idx))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    print(f"Using device: {device}")
    print(f"Testing on {len(dataset)} samples...")
    
    # Evaluation metrics
    total_samples = 0
    correct_fruit = 0
    correct_quality = 0
    correct_both = 0
    
    fruit_correct_per_class = {fruit: 0 for fruit in fruit2idx.keys()}
    fruit_total_per_class = {fruit: 0 for fruit in fruit2idx.keys()}
    quality_correct_per_class = {quality: 0 for quality in quality2idx.keys()}
    quality_total_per_class = {quality: 0 for quality in quality2idx.keys()}
    
    # Reverse mappings
    idx2fruit = {v: k for k, v in fruit2idx.items()}
    idx2quality = {v: k for k, v in quality2idx.items()}
    
    with torch.no_grad():
        for batch_idx, (images, fruit_labels, quality_labels) in enumerate(tqdm(dataloader, desc="Evaluating")):
            images = images.to(device)
            fruit_labels = fruit_labels.to(device)
            quality_labels = quality_labels.to(device)
            
            # Forward pass
            fruit_logits, quality_logits = model(images)
            
            # Get predictions
            fruit_pred = fruit_logits.argmax(dim=1)
            quality_pred = quality_logits.argmax(dim=1)
            
            # Calculate accuracy
            batch_size = images.size(0)
            total_samples += batch_size
            
            # Overall accuracy
            correct_fruit += (fruit_pred == fruit_labels).sum().item()
            correct_quality += (quality_pred == quality_labels).sum().item()
            correct_both += ((fruit_pred == fruit_labels) & (quality_pred == quality_labels)).sum().item()
            
            # Per-class accuracy
            for i in range(batch_size):
                fruit_true = fruit_labels[i].item()
                quality_true = quality_labels[i].item()
                fruit_name = idx2fruit[fruit_true]
                quality_name = idx2quality[quality_true]
                
                fruit_total_per_class[fruit_name] += 1
                quality_total_per_class[quality_name] += 1
                
                if fruit_pred[i].item() == fruit_true:
                    fruit_correct_per_class[fruit_name] += 1
                if quality_pred[i].item() == quality_true:
                    quality_correct_per_class[quality_name] += 1
    
    # Calculate final metrics
    fruit_accuracy = correct_fruit / total_samples * 100
    quality_accuracy = correct_quality / total_samples * 100
    both_accuracy = correct_both / total_samples * 100
    
    print("\n" + "="*60)
    print("📊 FINAL ACCURACY RESULTS")
    print("="*60)
    print(f"Total Samples Tested: {total_samples:,}")
    print(f"🍎 Fruit Classification Accuracy: {fruit_accuracy:.2f}%")
    print(f"📊 Quality Detection Accuracy: {quality_accuracy:.2f}%")
    print(f"🎯 Both Correct (Combined): {both_accuracy:.2f}%")
    
    print("\n📈 PER-CLASS FRUIT ACCURACY:")
    print("-" * 40)
    for fruit in sorted(fruit2idx.keys()):
        if fruit_total_per_class[fruit] > 0:
            acc = fruit_correct_per_class[fruit] / fruit_total_per_class[fruit] * 100
            print(f"{fruit:20s}: {acc:6.2f}% ({fruit_correct_per_class[fruit]}/{fruit_total_per_class[fruit]})")
    
    print("\n📊 PER-CLASS QUALITY ACCURACY:")
    print("-" * 40)
    for quality in sorted(quality2idx.keys()):
        if quality_total_per_class[quality] > 0:
            acc = quality_correct_per_class[quality] / quality_total_per_class[quality] * 100
            print(f"{quality:20s}: {acc:6.2f}% ({quality_correct_per_class[quality]}/{quality_total_per_class[quality]})")
    
    return {
        'fruit_accuracy': fruit_accuracy,
        'quality_accuracy': quality_accuracy,
        'both_accuracy': both_accuracy,
        'total_samples': total_samples,
        'fruit_per_class': fruit_correct_per_class,
        'quality_per_class': quality_correct_per_class
    }

if __name__ == "__main__":
    model_path = "fruitnet_multitask.pth"
    data_dir = "d:\\FruitNet-Project\\data"
    
    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint not found: {model_path}")
        exit(1)
    
    results = evaluate_model(model_path, data_dir)
    
    print("\n🎉 EVALUATION COMPLETE!")
    print(f"Your model achieved {results['fruit_accuracy']:.1f}% fruit accuracy")
    print(f"and {results['quality_accuracy']:.1f}% quality accuracy!")