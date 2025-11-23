# FruitNet Quick Setup 🚀

## Install Dependencies:
```bash
pip install torch-directml numpy Pillow opencv-python matplotlib tqdm
```

## Test Model:
```bash
python src/test_inference.py "your_fruit_image.jpg"
```

## Train Model:
```bash
python src/train.py --data-dir "data" --epochs 5
```

## Check Accuracy:
```bash
python src/evaluate_model.py
```

**Expected Results:**
- Fruit Classification: **93.21%**
- Quality Detection: **96.11%**