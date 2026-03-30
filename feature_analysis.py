import os
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

from models import MNIST_Net, CIFAR_Net


device = "cuda" if torch.cuda.is_available() else "cpu"

SAVE_DIR = "./results/feature_analysis"
os.makedirs(SAVE_DIR, exist_ok=True)


# Dataset
mnist_testset = torchvision.datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=T.ToTensor()
)

cifar_testset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=T.ToTensor()
)



# Load models
mnist_model = MNIST_Net().to(device)
mnist_model.load_state_dict(
    torch.load("./ckpt/mnist.pth", map_location=device)
)
mnist_model.eval()

cifar_model = CIFAR_Net().to(device)
cifar_model.load_state_dict(
    torch.load("./ckpt/cifar.pth", map_location=device)
)
cifar_model.eval()



# feature extractor
def get_features(model, x):
    # MNIST CNN
    if isinstance(model, MNIST_Net):
        x = model.pool(F.relu(model.conv1(x)))
        x = model.pool(F.relu(model.conv2(x)))
        x = torch.flatten(x, 1)
        feat = F.relu(model.fc1(x))
        return feat

    # CIFAR ResNet18
    elif isinstance(model, CIFAR_Net):
        m = model.model

        x = m.conv1(x)
        x = m.bn1(x)
        x = m.relu(x)

        x = m.layer1(x)
        x = m.layer2(x)
        x = m.layer3(x)
        x = m.layer4(x)

        x = m.avgpool(x)
        feat = torch.flatten(x, 1)
        return feat

    else:
        raise ValueError("Unsupported model type")



# Collect features
def collect_features(model, dataset, max_samples=2000):
    loader = DataLoader(dataset, batch_size=128, shuffle=False)

    feats, labels = [], []
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)

            feat = get_features(model, x)

            feats.append(feat.cpu())
            labels.append(y)

            total += x.size(0)
            if total >= max_samples:
                break

    feats = torch.cat(feats, dim=0)[:max_samples].numpy()
    labels = torch.cat(labels, dim=0)[:max_samples].numpy()

    return feats, labels



# PCA scatter
def plot_pca(features, labels, prefix):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(features)

    plt.figure(figsize=(8, 6))

    for cls in range(10):
        idx = labels == cls
        plt.scatter(
            reduced[idx, 0],
            reduced[idx, 1],
            s=10,
            alpha=0.6,
            label=str(cls)
        )

    plt.title(f"{prefix} PCA")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    plt.savefig(
        os.path.join(SAVE_DIR, f"{prefix.lower()}_pca.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()
    plt.close()


# Centroid distance heatmap
def plot_centroid_heatmap(features, labels, prefix):
    centroids = []

    for cls in range(10):
        centroids.append(features[labels == cls].mean(axis=0))

    centroids = np.stack(centroids)
    dist = pairwise_distances(centroids)

    plt.figure(figsize=(7, 6))
    sns.heatmap(dist, annot=True, fmt=".2f")
    plt.title(f"{prefix} Centroid Distance")
    plt.xlabel("Class")
    plt.ylabel("Class")
    plt.tight_layout()

    plt.savefig(
        os.path.join(SAVE_DIR, f"{prefix.lower()}_centroid_heatmap.png"),
        dpi=300
    )
    plt.show()
    plt.close()


# Feature norm histogram
def plot_feature_norm(features, labels, prefix):
    norms = np.linalg.norm(features, axis=1)

    plt.figure(figsize=(8, 5))
    for cls in range(10):
        cls_norm = norms[labels == cls]
        plt.hist(cls_norm, bins=30, alpha=0.4, label=str(cls))

    plt.title(f"{prefix} Feature Norm")
    plt.xlabel("L2 Norm")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(SAVE_DIR, f"{prefix.lower()}_feature_norm.png"),
        dpi=300
    )
    plt.show()
    plt.close()


# Logit margin histogram
def plot_logit_margin(model, dataset, prefix):
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    margins = []

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            logits = model(x)

            top2 = torch.topk(logits, 2, dim=1).values
            margin = (top2[:, 0] - top2[:, 1]).cpu().numpy()
            margins.extend(margin)

    plt.figure(figsize=(7, 5))
    plt.hist(margins, bins=50, alpha=0.7)
    plt.title(f"{prefix} Logit Margin")
    plt.xlabel("Top1 - Top2")
    plt.ylabel("Count")
    plt.tight_layout()

    plt.savefig(
        os.path.join(SAVE_DIR, f"{prefix.lower()}_logit_margin.png"),
        dpi=300
    )
    plt.show()
    plt.close()



def run_analysis(model, dataset, prefix):
    features, labels = collect_features(model, dataset)

    plot_pca(features, labels, prefix)
    plot_centroid_heatmap(features, labels, prefix)
    plot_feature_norm(features, labels, prefix)
    plot_logit_margin(model, dataset, prefix)


# Run
run_analysis(mnist_model, mnist_testset, "MNIST")
run_analysis(cifar_model, cifar_testset, "CIFAR10")