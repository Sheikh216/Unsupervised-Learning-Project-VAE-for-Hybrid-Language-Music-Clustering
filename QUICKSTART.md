# Quick Start Guide

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify installation:**
   ```bash
   python -c "import numpy, matplotlib, sklearn, tensorflow; print('All packages installed!')"
   ```

## Running Your First Model

### Option 1: Command Line (Recommended for Beginners)

**Train on MNIST with default settings:**
```bash
python main.py --dataset mnist
```

**Train with custom settings:**
```bash
python main.py --dataset mnist --architecture 256 128 --epochs 30 --learning-rate 0.05 --save-model
```

### Option 2: Run Examples Script

**Run all examples (demonstrates various features):**
```bash
python examples.py
```

This will:
- Train multiple models
- Show different optimizers
- Generate visualizations
- Save and load models

### Option 3: Python API (For Advanced Users)

Create a file `my_training.py`:

```python
from data_loader import DataLoader
from neural_network import NeuralNetwork
from trainer import Trainer
from visualizer import Visualizer

# 1. Load data
loader = DataLoader('mnist')
(X_train, y_train), (X_test, y_test) = loader.load_data()

# 2. Create model
model = NeuralNetwork(
    layer_sizes=[784, 128, 64, 10],
    activations=['relu', 'relu', 'softmax'],
    loss='categorical_crossentropy'
)

# 3. Train
trainer = Trainer(model, learning_rate=0.01, epochs=20)
history = trainer.train(X_train, y_train)

# 4. Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# 5. Visualize
viz = Visualizer()
viz.plot_training_history(history)
```

Then run:
```bash
python my_training.py
```

## Available Datasets

### MNIST (Handwritten Digits)
```bash
python main.py --dataset mnist
```
- **Best for:** Beginners, quick experiments
- **Expected accuracy:** ~97%
- **Training time:** ~5 minutes

### Fashion-MNIST (Clothing Items)
```bash
python main.py --dataset fashion_mnist
```
- **Best for:** Slightly harder classification
- **Expected accuracy:** ~88-90%
- **Training time:** ~5 minutes

### CIFAR-10 (Natural Images)
```bash
python main.py --dataset cifar10
```
- **Best for:** Challenging task
- **Expected accuracy:** ~45-55% (with simple network)
- **Training time:** ~10 minutes

## Common Configurations

### Quick Test (Fast Training)
```bash
python main.py --dataset mnist --architecture 64 --epochs 5 --batch-size 128
```

### High Accuracy (Slower but Better)
```bash
python main.py --dataset mnist --architecture 256 128 64 --epochs 30 --learning-rate 0.01 --early-stopping
```

### Deep Network
```bash
python main.py --dataset mnist --architecture 512 256 128 64 32 --epochs 25
```

### With Momentum Optimizer
```bash
python main.py --dataset mnist --optimizer momentum --learning-rate 0.05
```

## Understanding the Output

### During Training
```
Epoch 1/20: 100%|██████████| 1875/1875 [00:15<00:00, loss: 0.2543, acc: 0.9234]
Epoch 1/20 - 15.23s - loss: 0.2543 - acc: 0.9234 - val_loss: 0.1234 - val_acc: 0.9567
```

- **loss**: Training loss (lower is better)
- **acc**: Training accuracy (higher is better)
- **val_loss**: Validation loss
- **val_acc**: Validation accuracy

### Final Results
```
Test Loss: 0.1234
Test Accuracy: 0.9567 (95.67%)
```

## Output Files

All results are saved to `./results/` (or your `--output-dir`):

- `training_history.png`: Loss and accuracy curves
- `confusion_matrix.png`: Confusion matrix
- `sample_predictions.png`: Visual predictions
- `weight_distribution.png`: Weight histograms
- `*_model_*.pkl`: Saved model weights (if `--save-model`)

## Troubleshooting

### Problem: "ModuleNotFoundError"
**Solution:** Install requirements
```bash
pip install -r requirements.txt
```

### Problem: "Out of memory"
**Solution:** Reduce batch size
```bash
python main.py --batch-size 16
```

### Problem: "Poor accuracy"
**Solution:** Try these:
```bash
# Increase epochs
python main.py --epochs 30

# Try different architecture
python main.py --architecture 256 128 64

# Adjust learning rate
python main.py --learning-rate 0.001

# Use momentum
python main.py --optimizer momentum
```

### Problem: "Training is slow"
**Solution:** 
```bash
# Increase batch size
python main.py --batch-size 128

# Reduce epochs
python main.py --epochs 10

# Smaller architecture
python main.py --architecture 64 32
```

## Next Steps

1. **Experiment with architectures:** Try different layer sizes
2. **Test different datasets:** Compare MNIST vs Fashion-MNIST vs CIFAR-10
3. **Tune hyperparameters:** Learning rate, batch size, epochs
4. **Analyze results:** Look at confusion matrices and sample predictions
5. **Save your best model:** Use `--save-model` flag

## Examples to Try

### Example 1: Compare Architectures
```bash
# Shallow network
python main.py --dataset mnist --architecture 128 --epochs 20

# Deep network
python main.py --dataset mnist --architecture 256 128 64 32 --epochs 20
```

### Example 2: Learning Rate Impact
```bash
# Low learning rate
python main.py --dataset mnist --learning-rate 0.001 --epochs 30

# High learning rate
python main.py --dataset mnist --learning-rate 0.1 --epochs 30
```

### Example 3: Full Pipeline
```bash
python main.py --dataset fashion_mnist \
               --architecture 256 128 64 \
               --learning-rate 0.05 \
               --batch-size 64 \
               --epochs 25 \
               --optimizer momentum \
               --early-stopping \
               --patience 5 \
               --save-model \
               --visualize
```

## Tips for Best Results

1. **Start simple:** Begin with small architectures and few epochs
2. **Use validation:** Always monitor validation metrics
3. **Early stopping:** Use `--early-stopping` to prevent overfitting
4. **Visualize:** Always check the plots to understand training
5. **Save models:** Save good models with `--save-model`
6. **Experiment:** Try different configurations to learn

## Need Help?

Check the [README.md](README.md) for detailed documentation!

Happy training! 🚀
