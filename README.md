# FruitNet Multi-Task Classifier 🍎🤖

An advanced deep learning model for **fruit type classification** and **quality assessment** using PyTorch with DirectML GPU acceleration.

## 🎯 Model Performance
- **🍎 Fruit Classification Accuracy: 93.21%**
- **📊 Quality Detection Accuracy: 96.11%** 
- **🎯 Combined Accuracy: 92.23%**
- **📈 Dataset Size: 19,555+ fruit images**

## 🚀 Features
- **Multi-task Learning**: Simultaneous fruit type and quality classification
- **GPU Acceleration**: DirectML support for fast training/inference
- **Data Augmentation**: Advanced preprocessing for better generalization
- **Real-time Inference**: Quick predictions on new fruit images
- **Production Ready**: Optimized model with high accuracy

## 📁 Project Structure
```
FruitNet-Project/
├── src/
│   ├── model.py           # CNN architecture with dual heads
│   ├── dataset.py         # Data loading with augmentation
│   ├── train.py           # Training script with optimizations
│   ├── test_inference.py  # Single image prediction
│   ├── evaluate_model.py  # Full dataset evaluation
│   ├── utils.py           # Helper functions
│   └── requirements.txt   # Dependencies
├── data/                  # Dataset (structure below)
└── fruitnet_multitask.pth # Trained model checkpoint
```

## 📊 Supported Categories

### Fruits (19 varieties):
- Apple, Apple_Good, Apple_Bad
- Banana, Banana_Good, Banana_Bad  
- Guava, Guava_Good, Guava_Bad
- Lime_Good, Lime_Bad
- Orange, Orange_Good, Orange_Bad
- Pomegranate, Pomegranate_Good, Pomegranate_Bad
- Lemon

### Quality Levels:
- **Good Quality_Fruits** (97.42% accuracy)
- **Bad Quality_Fruits** (94.76% accuracy)
- **Mixed Quality_Fruits** (90.57% accuracy)

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Shrey3satdeve/fruitnet-project.git
cd fruitnet-project
```

2. **Create virtual environment:**
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
```

3. **Install dependencies:**
```bash
pip install -r src/requirements.txt
```

## 🔧 Usage

### Quick Test (Single Image):
```bash
python src/test_inference.py "path/to/fruit_image.jpg"
```

### Train Model:
```bash
python src/train.py --data-dir "data" --epochs 5 --batch-size 32
```

### Evaluate Model:
```bash
python src/evaluate_model.py
```

### Check Dataset:
```bash
python src/dataset.py "data"
```

## 📈 Model Architecture

- **Backbone**: 3 Convolutional layers with BatchNorm and MaxPooling
- **Shared Features**: AdaptiveAvgPool2d + Fully Connected layers
- **Dual Heads**: Separate classifiers for fruit type and quality
- **Optimization**: Adam optimizer with StepLR scheduling
- **Regularization**: Dropout and data augmentation

## 🎯 Training Results

```
Final Training Metrics:
├── Fruit Classification: 93.21%
├── Quality Detection: 96.11% 
├── Combined Accuracy: 92.23%
└── Dataset: 19,555 samples
```

### Top Performing Categories:
- Pomegranate_Good: 97.88%
- Lime_Bad: 96.77%
- Lime_Good: 96.44%
- Banana_Bad: 94.11%

## 🔬 Technical Highlights

- **Multi-task Learning**: Single model, dual outputs
- **Data Augmentation**: Random crops, flips, rotations, color jitter
- **GPU Acceleration**: DirectML for Windows GPU support
- **Learning Rate Scheduling**: Adaptive learning rate decay
- **Robust Evaluation**: Per-class accuracy metrics

## 📱 Real-World Applications

- 🏪 **Grocery Stores**: Automatic fruit quality inspection
- 🏭 **Food Industry**: Quality control in processing
- 🌾 **Agriculture**: Harvest assessment and grading
- 🛒 **E-commerce**: Product quality verification



## 📊 Dataset

### **Download Dataset:**
- **📁 Size:** 19,555 fruit images (~3.2 GB)
- **📂 Format:** JPEG images organized in hierarchical folders
- **🔗 Download Link:** [Contact for Dataset Access](https://www.kaggle.com/datasets/shashwatwork/fruitnet-indian-fruits-dataset-with-quality?select=Processed+Images_Fruits)
- **☁️ Alternative:** Available on request via Google Drive/OneDrive

### **Dataset Structure:**
```
data/
├── Good_Quality_Fruits/
│   ├── Apple_Good/           # 1,149 images
│   ├── Banana_Good/          # 1,113 images  
│   ├── Guava_Good/           # 1,152 images
│   ├── Lime_Good/            # 1,094 images
│   ├── Orange_Good/          # 1,216 images
│   └── Pomegranate_Good/     # 5,940 images
├── Bad_Quality_Fruits/
│   ├── Apple_Bad/            # 1,141 images
│   ├── Banana_Bad/           # 1,087 images
│   ├── Guava_Bad/            # 1,129 images
│   ├── Lime_Bad/             # 1,085 images
│   ├── Orange_Bad/           # 1,159 images
│   └── Pomegranate_Bad/      # 1,187 images
└── Mixed_Quality_Fruits/
    ├── Apple/                # 113 images
    ├── Banana/               # 285 images
    ├── Guava/                # 148 images
    ├── Lemon/                # 278 images
    ├── Orange/               # 125 images
    └── Pomegranate/          # 125 images
```


## 🛡️ Requirements

- Python 3.8+
- PyTorch 2.0+
- torch-directml (for GPU)
- OpenCV, Pillow, NumPy
- tqdm, matplotlib

## 🚀 Performance Optimizations

- ✅ Data augmentation for better generalization
- ✅ Learning rate scheduling for optimal convergence  
- ✅ Batch size optimization for memory efficiency
- ✅ GPU acceleration with DirectML
- ✅ Early stopping to prevent overfitting

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

📺 **Live Demo Video:** [Watch FruitNet AI in Action](https://docs.google.com/videos/d/16pUqNu5ZnT16TzpDkwDq__HbjIECLHM5FmSxeoUGwuY/edit?usp=sharing)

## 📧 Contact

**Contact Author**: Shreyash Satadeve  
**Email**: shreyashsatadeve@gmail.com  
**GitHub**: [@Shrey3satdeve](https://github.com/Shrey3satdeve)  
**Project Repository**: [fruitnet-project](https://github.com/Shrey3satdeve/fruitnet-project)

---

⭐ **If you found this project helpful, please give it a star!** ⭐
