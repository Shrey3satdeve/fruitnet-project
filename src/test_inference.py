try:
	# Prefer explicit ModuleNotFoundError to avoid masking other import errors
	import torch
except ModuleNotFoundError:
	print("Error: Python package 'torch' is not installed.")
	print("Install PyTorch for your platform. For a quick CPU-only install, run:")
	print("  pip install --upgrade pip")
	print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
	print("For CUDA support, follow instructions at https://pytorch.org/get-started/locally/")
	raise SystemExit(1)

# Try to use DirectML if available, otherwise fall back to CPU
try:
	import torch_directml as dml
except (ModuleNotFoundError, ImportError):
	dml = None
	print("Warning: 'torch-directml' not installed or failed to load. Falling back to CPU. To enable DirectML/GPU, run:")
	print("  pip install torch-directml")
	# continue with CPU device

if dml is not None:
	device = dml.device()
else:
	device = torch.device("cpu")
	print(f"Using device: {device}")

import os
import sys
import glob
from PIL import Image
import torchvision.transforms as T
from model import FruitNetMultiTask

# Load checkpoint first to get label mappings and head sizes
checkpoint_path = "fruitnet_multitask.pth"
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

# Try loading with weights_only=True when available to reduce pickle surface.
# If that fails (e.g. UnpicklingError), fall back to the full load with a warning.
try:
	checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
except TypeError:
	# older PyTorch doesn't support weights_only kwarg
	checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
except Exception as e:
	# weights_only may reject some objects; fall back to full load
	print(f"Warning: weights-only load failed ({e}). Falling back to full torch.load().")
	print("Only do this if you trust the checkpoint file, as it may execute arbitrary code during unpickling.")
	checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

# Validate mappings exist in checkpoint
if "fruit2idx" not in checkpoint or "quality2idx" not in checkpoint:
    raise KeyError("Checkpoint missing 'fruit2idx' or 'quality2idx' mappings. Cannot determine head sizes.")

num_fruit = len(checkpoint["fruit2idx"])
num_quality = len(checkpoint["quality2idx"])

# Build model with sizes that match the checkpoint
model = FruitNetMultiTask(num_fruit_classes=num_fruit, num_quality_classes=num_quality)
model.load_state_dict(checkpoint["model_state_dict"])  # will match now
model.to(device)
model.eval()

# Label mappings (index -> label)
fruit_idx2label = {v: k for k, v in checkpoint["fruit2idx"].items()}
quality_idx2label = {v: k for k, v in checkpoint["quality2idx"].items()}

# Transform (match training preprocessing: Resize, ToTensor, Normalize)
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Test Image: allow CLI override or find any image in repo
if len(sys.argv) > 1:
	img_path = sys.argv[1]
else:
	img_path = "test.jpg"

if not os.path.exists(img_path):
	# search current dir and top-level subdirectories for an image
	candidates = []
	exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
	for ext in exts:
		candidates.extend(glob.glob(ext))
		for d in os.listdir('.'):
			if os.path.isdir(d):
				candidates.extend(glob.glob(os.path.join(d, ext)))
	if candidates:
		img_path = candidates[0]
		print(f"Using found image: {img_path}")
	else:
		raise FileNotFoundError(f"Test image not found: {img_path}. Provide a path as `python test_inference.py <image>`")

img = Image.open(img_path).convert("RGB")

inp = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    fruit_logits, quality_logits = model(inp)

fruit_class = int(fruit_logits.argmax().item())
quality_class = int(quality_logits.argmax().item())

print("Predicted Fruit:", fruit_idx2label.get(fruit_class, str(fruit_class)))
print("Predicted Quality:", quality_idx2label.get(quality_class, str(quality_class)))
