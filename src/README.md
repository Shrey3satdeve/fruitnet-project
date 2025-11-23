# FruitNet Multi-task (Fruit type + Quality)

Python 3.10 required.

Install dependencies:

```powershell
pip install torch-directml numpy pillow opencv-python matplotlib tqdm
```

Usage (train):

```powershell
python train.py --data-dir "D:\\archive\\Processed Images_Fruits\\Mixed Qualit_Fruits" --epochs 10 --batch-size 32
```

Notes:
- Uses `torch_directml` as the device backend: `device = dml.device()` (no CUDA).
- Dataset structure expected:

```
base_dir/
  <quality_name>/    # e.g. Good, Bad, Mixed
    <fruit_name>/     # e.g. Apple, Banana, ...
      image files
```

- Outputs saved to `fruitnet_multitask.pth` (contains model state dict and label mappings).
