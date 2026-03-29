# adversarial-fgsm-pgd


## 📂 Project Structure
.
├── ckpt/ # saved model checkpoints
├── results/ # generated results (images, logs, etc.)
├── attack.py # FGSM / PGD implementations
├── models.py # MNIST / CIFAR-10 models
├── train.py # model training script
├── test.py # evaluation / attack script
├── analyze.ipynb # analysis notebook
├── visualize.ipynb # visualization notebook
├── attack_success_rate.json # attack performance results
├── requirements.txt
├── .gitignore
└── README.md

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