# 🚀 GETTING STARTED - Complete Guide

## Step-by-Step Installation & First Run

### Step 1: Install Dependencies

**Windows:**
```bash
setup.bat
```

**Linux/Mac:**
```bash
chmod +x setup.sh
./setup.sh
```

**Manual Installation:**
```bash
pip install -r requirements.txt
```

### Step 2: Run Your First Training

**Option A: Quick Test (5 minutes)**
```bash
python main.py --dataset mnist --epochs 5
```

**Option B: Full Training (Recommended)**
```bash
python main.py --dataset mnist
```

**Option C: Run All Examples**
```bash
python examples.py
```

### Step 3: View Results

After training completes, check the `results/` folder for:
- Training curves
- Confusion matrix
- Sample predictions
- Weight distributions

---

## 🎯 Quick Commands Reference

### Basic Training Commands

| Command | Description | Time |
|---------|-------------|------|
| `python main.py --dataset mnist` | Train on MNIST (default) | ~5 min |
| `python main.py --dataset fashion_mnist` | Train on Fashion-MNIST | ~5 min |
| `python main.py --dataset cifar10` | Train on CIFAR-10 | ~10 min |
| `python examples.py` | Run all examples | ~15 min |

### Customization Examples

**Change Architecture:**
```bash
python main.py --dataset mnist --architecture 256 128 64
```

**Adjust Learning Rate:**
```bash
python main.py --dataset mnist --learning-rate 0.001
```

**More Epochs:**
```bash
python main.py --dataset mnist --epochs 50
```

**Use Momentum:**
```bash
python main.py --dataset mnist --optimizer momentum
```

**Enable Early Stopping:**
```bash
python main.py --dataset mnist --early-stopping --patience 5
```

**Save Model:**
```bash
python main.py --dataset mnist --save-model
```

**Combined (Best Settings):**
```bash
python main.py --dataset mnist --architecture 256 128 64 --learning-rate 0.01 --epochs 30 --optimizer momentum --early-stopping --save-model
```

---

## 📊 What Happens During Training

```
Loading mnist dataset...
Dataset loaded: 60000 training samples, 10000 test samples
Images flattened to shape: (784,)
Labels one-hot encoded: 10 classes

Creating neural network with architecture: [128, 64]

Training model...
  Learning rate: 0.01
  Batch size: 32
  Epochs: 20
  Optimizer: sgd

Epoch 1/20: 100%|███████| 1875/1875 [00:15<00:00, loss: 0.2543, acc: 0.9234]
Epoch 1/20 - 15.23s - loss: 0.2543 - acc: 0.9234 - val_loss: 0.1234 - val_acc: 0.9567
Epoch 2/20: 100%|███████| 1875/1875 [00:14<00:00, loss: 0.1234, acc: 0.9567]
...

EVALUATION ON TEST SET
Test Loss: 0.0987
Test Accuracy: 0.9678 (96.78%)

GENERATING VISUALIZATIONS
Training history plot saved to results/training_history.png
Confusion matrix saved to results/confusion_matrix.png
Sample predictions saved to results/sample_predictions.png
```

---

## 🗂️ File Descriptions

### Core Files (What Each Does)

| File | Purpose | Lines |
|------|---------|-------|
| `main.py` | Main training script | 356 |
| `neural_network.py` | Neural network implementation | 309 |
| `trainer.py` | Training loop and optimization | 222 |
| `data_loader.py` | Dataset loading | 169 |
| `activations.py` | Activation functions | 134 |
| `losses.py` | Loss functions | 127 |
| `visualizer.py` | Visualization tools | 311 |
| `metrics.py` | Evaluation metrics | 150 |
| `examples.py` | Working examples | 372 |

### Documentation Files

| File | Purpose |
|------|---------|
| `README.md` | Complete documentation |
| `QUICKSTART.md` | Quick start guide |
| `PROJECT_SUMMARY.md` | Project overview |
| `GETTING_STARTED.md` | This file! |

### Configuration Files

| File | Purpose |
|------|---------|
| `requirements.txt` | Python dependencies |
| `.gitignore` | Git ignore rules |
| `setup.bat` | Windows setup script |
| `setup.sh` | Linux/Mac setup script |

---

## 🎓 Learning Path

### Beginner (Day 1)
1. Run setup script
2. Run: `python main.py --dataset mnist`
3. Check `results/` folder
4. Read QUICKSTART.md

### Intermediate (Day 2-3)
1. Run: `python examples.py`
2. Try different datasets
3. Experiment with architectures
4. Read README.md

### Advanced (Week 1)
1. Modify activation functions
2. Implement new optimizers
3. Add regularization
4. Create custom datasets

---

## 🔍 Troubleshooting

### Issue: Installation Fails

**Solution:**
```bash
# Update pip first
python -m pip install --upgrade pip

# Install packages one by one
pip install numpy
pip install matplotlib
pip install scikit-learn
pip install tensorflow
pip install keras
pip install tqdm
pip install seaborn
pip install pandas
pip install pillow
```

### Issue: "No module named 'activations'"

**Solution:** Make sure you're in the project directory
```bash
cd c:\Users\USERAS\Desktop\715_Project
python main.py --dataset mnist
```

### Issue: Training is too slow

**Solution 1:** Increase batch size
```bash
python main.py --batch-size 128
```

**Solution 2:** Reduce epochs
```bash
python main.py --epochs 10
```

**Solution 3:** Use smaller architecture
```bash
python main.py --architecture 64 32
```

### Issue: Low accuracy

**Solution 1:** Train longer
```bash
python main.py --epochs 30
```

**Solution 2:** Adjust learning rate
```bash
python main.py --learning-rate 0.001
```

**Solution 3:** Try momentum
```bash
python main.py --optimizer momentum
```

---

## 📈 Expected Results

### MNIST
- **Quick test (5 epochs):** ~95% accuracy
- **Full training (20 epochs):** ~97% accuracy
- **Optimized (30 epochs + momentum):** ~98% accuracy

### Fashion-MNIST
- **Quick test (5 epochs):** ~82% accuracy
- **Full training (20 epochs):** ~88% accuracy
- **Optimized (30 epochs + momentum):** ~90% accuracy

### CIFAR-10
- **Quick test (5 epochs):** ~35% accuracy
- **Full training (20 epochs):** ~50% accuracy
- **Optimized (30 epochs + deeper network):** ~55% accuracy

---

## 🎯 Common Use Cases

### Use Case 1: Quick Experiment
```bash
python main.py --dataset mnist --epochs 5 --batch-size 128
```
**When:** Testing code changes, quick verification
**Time:** 2-3 minutes

### Use Case 2: Production Training
```bash
python main.py --dataset mnist --architecture 256 128 64 --epochs 30 --early-stopping --save-model
```
**When:** Training a model for actual use
**Time:** 10-15 minutes

### Use Case 3: Hyperparameter Tuning
```bash
# Test different learning rates
python main.py --learning-rate 0.001 --epochs 20
python main.py --learning-rate 0.01 --epochs 20
python main.py --learning-rate 0.1 --epochs 20
```
**When:** Finding best hyperparameters
**Time:** 30 minutes

### Use Case 4: Architecture Search
```bash
# Try different architectures
python main.py --architecture 128 --epochs 15
python main.py --architecture 256 128 --epochs 15
python main.py --architecture 512 256 128 --epochs 15
```
**When:** Finding best architecture
**Time:** 45 minutes

---

## 💡 Tips for Success

1. **Start Simple:** Begin with MNIST and default settings
2. **Monitor Training:** Watch the progress bars and metrics
3. **Check Visualizations:** Always look at the plots
4. **Save Good Models:** Use `--save-model` for best results
5. **Experiment:** Try different configurations
6. **Be Patient:** Deep learning takes time!

---

## 🎊 Success Checklist

- [ ] Dependencies installed
- [ ] First training completed
- [ ] Results folder created
- [ ] Visualizations generated
- [ ] Examples script runs
- [ ] Understanding basic usage
- [ ] Ready to experiment!

---

## 📞 Next Steps

1. **Read Documentation:**
   - QUICKSTART.md for quick reference
   - README.md for detailed info
   - PROJECT_SUMMARY.md for overview

2. **Run Examples:**
   ```bash
   python examples.py
   ```

3. **Experiment:**
   - Try different datasets
   - Modify architectures
   - Tune hyperparameters

4. **Learn More:**
   - Read the code
   - Understand backpropagation
   - Implement new features

---

**You're ready to start! 🚀**

Run this command to begin:
```bash
python main.py --dataset mnist
```

Good luck and happy learning!
