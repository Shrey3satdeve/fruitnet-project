# Example Images

This folder contains sample images for demonstration purposes.

## Dataset Access

The full dataset (19,555 images) is available on request due to size limitations.

**Contact:** shreyashsatadeve@gmail.com

## Sample Predictions

You can test the model with any fruit image using:

```bash
python src/test_inference.py "path/to/your/image.jpg"
```

Expected output:
```
Predicted Fruit: Apple_Good
Predicted Quality: Good Quality_Fruits
```

## Dataset Statistics

- **Total Images:** 19,555
- **Fruit Types:** 19 varieties
- **Quality Levels:** 3 categories
- **Format:** JPEG images (224x224 after preprocessing)
- **Size:** ~3.2 GB total