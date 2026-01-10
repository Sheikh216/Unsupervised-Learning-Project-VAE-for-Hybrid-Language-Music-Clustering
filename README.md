<<<<<<< HEAD
# Unsupervised-Learning-Project-VAE-for-Hybrid-Language-Music-Clustering
=======
# Neural Network from Scratch

A complete implementation of a neural network from scratch using NumPy. This project demonstrates the fundamentals of deep learning by building a fully functional feedforward neural network without relying on high-level deep learning frameworks.

## � **NEW: Music & Audio Classification Support!**

This project now supports **TWO types of datasets**:
1. **Image Classification** (MNIST, Fashion-MNIST, CIFAR-10) - Use `main.py`
2. **Music Classification** (GTZAN, MSD, Jamendo) - Use `main_music.py` ⭐ NEW!

See [README_MUSIC.md](README_MUSIC.md) for music dataset documentation.

## �🌟 Features

- **Pure NumPy Implementation**: Built entirely with NumPy for educational purposes
- **Multiple Activation Functions**: ReLU, Sigmoid, Tanh, Softmax, Leaky ReLU
- **Various Loss Functions**: MSE, Binary Cross-Entropy, Categorical Cross-Entropy, Huber Loss
- **Optimizers**: SGD, Momentum
- **Automatic Dataset Loading**: 
  - **Images**: MNIST, Fashion-MNIST, CIFAR-10
  - **Music**: GTZAN, Million Song Dataset, Jamendo (with sample data fallback)
- **Audio Feature Extraction**: MFCCs, Spectral features, Rhythm features (43 features total)
- **Training Utilities**: Batch processing, validation splits, early stopping
- **Comprehensive Visualizations**: Training curves, confusion matrices, sample predictions
- **Model Persistence**: Save and load trained models

## 📋 Requirements

Install the required packages using:

```bash
pip install -r requirements.txt
```

### Dependencies
- numpy >= 1.21.0
- matplotlib >= 3.4.0
- scikit-learn >= 1.0.0
- pandas >= 1.3.0
- seaborn >= 0.11.0
- tensorflow >= 2.8.0 (for dataset loading only)
- keras >= 2.8.0 (for dataset loading only)
- pillow >= 8.0.0
- tqdm >= 4.62.0
- **librosa >= 0.9.0** (for music/audio datasets)
- **soundfile >= 0.11.0** (for audio I/O)
- **audioread >= 2.1.9** (for audio decoding)
Image Classification (Original)

Train a neural network on MNIST with default settings:

```bash
python main.py --dataset mnist
```

### 🎵 Music Classification (NEW!)

Train a neural network on music genres (uses sample data if dataset not downloaded):

```bash
python main_music.py --dataset gtzan
```

See [README_MUSIC.md](README_MUSIC.md) for complete music dataset documentation!
```bash
python main.py --dataset mnist
```

### Advanced Usage

Train with custom architecture and hyperparameters:

```bash
python main.py --dataset mnist \
               --architecture 256 128 64 \
               --learning-rate 0.01 \
               --batch-size 64 \
               --epochs 30 \
               --optimizer momentum \
               --early-stopping \
               --save-model
```

### Available Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | mnist | Dataset to use (mnist, fashion_mnist, cifar10) |
| `--architecture` | 128 64 | Hidden layer sizes (space-separated) |
| `--learning-rate` | 0.01 | Learning rate for optimization |
| `--batch-size` | 32 | Batch size for training |
| `--epochs` | 20 | Number of training epochs |
| `--optimizer` | sgd | Optimizer (sgd, momentum) |
| `--early-stopping` | False | Image classification training script
├── main_music.py            # 🎵 Music classification training script (NEW!)
├── neural_network.py        # Neural network implementation
├── activations.py           # Activation functions
├── losses.py                # Loss functions
├── trainer.py               # Training utilities
├── data_loader.py           # Image dataset loading
├── audio_data_loader.py     # 🎵 Audio dataset loading (NEW!)
├── visualizer.py            # Visualization utilities
├── metrics.py               # Evaluation metrics
├── examples.py              # Working examples
├── requirements.txt         # Project dependencies
├── README.md                # This file (image datasets)
├── README_MUSIC.md          # 🎵 Music datasets documentation (NEW!)
├── QUICKSTART.md            # Quick start guide
├── .gitignore               # Git ignore file
└── setup.bat/setup.sh       # Setup scripts
├── main.py                  # Main training script
├── neural_network.py        # Neural network implementation
├── activations.py           # Activation functions
├── losses.py                # Loss functions
├── trainer.py               # Training utilities
├── data_loader.py           # Dataset loading and preprocessing
├── visualizer.py            # Visualization utilities
├── requirements.txt         # Project dependencies
├── .gitignore              # Git ignore file
└── README.md               # This file
```

## 🧠 Architecture

### Neural Network Components

1. **Layer Class**: Implements a fully connected (dense) layer
   - Forward propagation with activation
   - Backward propagation for gradient computation
   - Weight initialization (He, Xavier, Random)

2. **NeuralNetwork Class**: Manages the entire network
   - Multiple layer support
   - Forward and backward propagation
   - Model evaluation and prediction
   - Model saving/loading

3. **Activation Functions**:
   - ReLU (Rectified Linear Unit)
   - Sigmoid
   - Tanh (Hyperbolic Tangent)
   - Softmax (for multi-class classification)
   - Leaky ReLU
   - Linear (Identity)

4. **Loss Functions**:
   - Mean Squared Error (MSE)
   - Binary Cross-Entropy
   - Categorical Cross-Entropy
   - Huber Loss
Image Datasets (via `main.py`)

#### 1. MNIST
- **Description**: Handwritten digits (0-9)
- **Size**: 60,000 training + 10,000 test images
- **Image Size**: 28×28 grayscale
- **Classes**: 10

#### 2. Fashion-MNIST
- **Description**: Fashion product images
- **Size**: 60,000 training + 10,000 test images
- **Image Size**: 28×28 grayscale
- **Classes**: 10 (T-shirt, Trouser, Pullover, etc.)

#### 3. CIFAR-10
- **Description**: Natural images
- **Size**: 50,000 training + 10,000 test images
- **Image Size**: 32×32 RGB
- **Classes**: 10 (Airplane, Car, Bird, etc.)

### 🎵 Music/Audio Datasets (via `main_music.py`) - NEW!

#### 1. GTZAN Genre Collection
- **Description**: Music genre classification
- **Size**: 1,000 audio tracks (100 per genre)
- **Duration**: 30 seconds each
- **Classes**: 10 (Blues, Classical, Country, Disco, Hip-Hop, Jazz, Metal, Pop, Reggae, Rock)
- **Features**: 43 audio features (MFCCs, spectral, rhythm)
- **Download**: http://marsyas.info/downloads/datasets.html
- **Note**: Sample data available if not downloaded

#### 2. Million Song Dataset (MSD)
- **Description**: Large-scale music dataset
- **Features**: Pre-computed audio features
- **Download**: http://millionsongdataset.com/

#### 3. Jamendo Dataset
- **Description**: Songs with audio + metadata + lyrics
- **Download**: https://www.kaggle.com/datasets/andradaolteanu/jamendo-music-dataset

#### 4. MIR-1K, Lakh MIDI, Lyrics Datasets
- See [README_MUSIC.md](README_MUSIC.md) for complete list and links
- **Classes**: 10 (T-shirt, Trouser, Pullover, etc.)

### 3. CIFAR-10
- **Description**: Natural images
- **Size**: 50,000 training + 10,000 test images
- **Image Size**: 32×32 RGB
- **Classes**: 10 (Airplane, Car, Bird, etc.)

## 🎯 Example Results

### MNIST Training Example

```bash
python main.py --dataset mnist --architecture 128 64 --epochs 20 --save-model
```

Expected performance:
- **Training Accuracy**: ~98%
- **Test Accuracy**: ~97%
- **Training Time**: ~5-10 minutes (CPU)

### Fashion-MNIST Training Example

```bash
python main.py --dataset fashion_mnist --architecture 256 128 64 --epochs 25
```

Expected performance:
- **Training Accuracy**: ~90-92%
- **Test Accuracy**: ~88-90%

## 📈 Visualizations

The project automatically generates the following visualizations:

1. **Training History**: Loss and accuracy curves over epochs
2. **Confusion Matrix**: Shows prediction performance per class
3. **Sample Predictions**: Visual comparison of predictions vs. true labels
4. **Weight Distributions**: Histogram of weights in each layer

All visualizations are saved to the `--output-dir` (default: `./results`).

## 🔧 Customization

### Using the API Directly

```python
from neural_network import NeuralNetwork
from data_loader import DataLoader
from trainer import Trainer

# Load data
loader = DataLoader('mnist')
(X_train, y_train), (X_test, y_test) = loader.load_data()

# Create model
model = NeuralNetwork(
    layer_sizes=[784, 128, 64, 10],
    activations=['relu', 'relu', 'softmax'],
    loss='categorical_crossentropy'
)

# Train
trainer = Trainer(model, learning_rate=0.01, batch_size=32, epochs=20)
history = trainer.train(X_train, y_train)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")
```

### Creating Custom Architectures

```python
# Simple network
model = NeuralNetwork(layer_sizes=[784, 10])

# Deep network
model = NeuralNetwork(layer_sizes=[784, 512, 256, 128, 64, 10])

# With custom activations
model = NeuralNetwork(
    layer_sizes=[784, 128, 64, 10],
    activations=['tanh', 'relu', 'softmax']
)
```

## 🧪 Testing Individual Modules

Each module can be tested independently:

```bash
# Test data loader
python data_loader.py

# Test neural network
python neural_network.py

# Test trainer
python trainer.py
```

## 📚 Educational Value

This project is designed for learning purposes and demonstrates:

1. **Forward Propagation**: How data flows through the network
2. **Backpropagation**: Gradient computation and chain rule
3. **Optimization**: Weight updates using gradients
4. **Batch Processing**: Efficient mini-batch training
5. **Regularization**: Through weight initialization
6. **Model Evaluation**: Metrics and visualization

## 🎓 How It Works

### Forward Propagation
For each layer $l$:
$$z^{[l]} = W^{[l]}a^{[l-1]} + b^{[l]}$$
$$a^{[l]} = g^{[l]}(z^{[l]})$$

where $g^{[l]}$ is the activation function.

### Backpropagation
Compute gradients:
$$\frac{\partial L}{\partial W^{[l]}} = \frac{1}{m} \delta^{[l]} (a^{[l-1]})^T$$
$$\frac{\partial L}{\partial b^{[l]}} = \frac{1}{m} \sum_i \delta^{[l]}_i$$

### Weight Update (SGD)
$$W^{[l]} := W^{[l]} - \alpha \frac{\partial L}{\partial W^{[l]}}$$

## 🐛 Troubleshooting

### Common Issues

1. **Memory Error**: Reduce batch size
   ```bash
   python main.py --batch-size 16
   ```

2. **Poor Performance**: Try different architecture or learning rate
   ```bash
   python main.py --architecture 256 128 64 --learning-rate 0.001
   ```

3. **Overfitting**: Use early stopping
   ```bash
   python main.py --early-stopping --patience 5
   ```

## 📝 License

This project is provided as-is for educational purposes.

## 🤝 Contributing

This is an educational project. Feel free to fork and modify for your learning!

## 📧 Contact

For questions or suggestions about this project, please open an issue.

## 🙏 Acknowledgments

- Datasets provided by Keras/TensorFlow
- Inspired by classic neural network implementations
- Built for deep learning education

---

**Happy Learning! 🚀**
>>>>>>> faf63be (Initial commit)
