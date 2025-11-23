import torch
from model import FruitNetMultiTask

# Define simple label mappings matching project data
fruit2idx = {'Apple': 0, 'Banana': 1, 'Guava': 2, 'Lemon': 3, 'Orange': 4, 'Pomegranate': 5}
quality2idx = {'Good': 0, 'Bad': 1, 'Mixed': 2}

num_fruit = len(fruit2idx)
num_quality = len(quality2idx)

model = FruitNetMultiTask(num_fruit_classes=num_fruit, num_quality_classes=num_quality)
state = {
    'model_state_dict': model.state_dict(),
    'fruit2idx': fruit2idx,
    'quality2idx': quality2idx,
    'epoch': 0,
}

save_path = 'fruitnet_multitask.pth'
torch.save(state, save_path)
print(f"Saved dummy checkpoint to {save_path} (num_fruit={num_fruit}, num_quality={num_quality})")
