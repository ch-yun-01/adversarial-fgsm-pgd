# adversarial-fgsm-pgd


## 📂 Project Structure
```bash
.
├── ckpt/                         # saved model checkpoints
├── results/                      # adversarial examples & attack success rate json
├── attack.py                     # FGSM / PGD implementations
├── models.py                     # MNIST / CIFAR-10 CNN architectures
├── train.py                      # baseline model training script
├── test.py                       # evaluation & attack execution script
├── analyze.ipynb                 # result analysis (JSON-based)
├── visualize.ipynb               # adversarial example visualization
├── requirements.txt              # dependencies
└── README.md
```

## Run attack / evaluation
python test.py

### 📊 Datasets
- **MNIST** (grayscale, 28×28)
- **CIFAR-10** (RGB, 32×32)

### ⚔️ Attacks
- **FGSM**
  - targeted
  - untargeted
- **PGD**
  - targeted
  - untargeted

### 📉 Loss Functions
- **Cross Entropy** (`ce`)
- **Mean Squared Error** (`mse`)
- **KL Divergence** (`kl`)
- **Margin Loss** (`margin`)
