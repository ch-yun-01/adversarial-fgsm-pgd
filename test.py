import torch
import torchvision
import torchvision.transforms as transforms

from models import MNIST_Net, CIFAR_Net
from train import train
from attack import fgsm_targeted, fgsm_untargeted, pgd_targeted

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# DATA
# =========================
mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

cifar_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

mnist_train = torchvision.datasets.MNIST('./data', train=True, download=True, transform=mnist_transform)
mnist_test = torchvision.datasets.MNIST('./data', train=False, download=True, transform=mnist_transform)

cifar_train = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=cifar_transform)
cifar_test = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=cifar_transform)

mnist_trainloader = torch.utils.data.DataLoader(mnist_train, batch_size=64, shuffle=True)
mnist_testloader = torch.utils.data.DataLoader(mnist_test, batch_size=64, shuffle=False)

cifar_trainloader = torch.utils.data.DataLoader(cifar_train, batch_size=64, shuffle=True)
cifar_testloader = torch.utils.data.DataLoader(cifar_test, batch_size=64, shuffle=False)


# TRAIN / LOAD
print("[TRAIN start] MNIST")
mnist_model = train(MNIST_Net(), mnist_trainloader, device, ckpt_path="./ckpt/mnist.pth")
print("[TRAIN done]")

print("[TRAIN start] CIFAR10")
cifar_model = train(CIFAR_Net(), cifar_trainloader, device, epochs = 10, ckpt_path="./ckpt/cifar.pth")
print("[TRAIN done]")


# CLEAN ACCURACY
def evaluate_model(model, loader):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            out = model(x)
            pred = out.argmax(1)

            correct += (pred == y).sum().item()
            total += y.size(0)

    return correct / total


# ATTACK ACCURACY
def evaluate_attack(model, loader, attack, targeted=False, eps=0.1):
    model.eval()

    correct = 0
    total = 0

    for i, (x, y) in enumerate(loader):
        if i >= 100:  # 속도용
            break

        x, y = x.to(device), y.to(device)

        if targeted:
            target = (y + 1) % 10
            adv = attack(model, x, target, eps)
            pred = model(adv).argmax(1)
            correct += (pred == target).sum().item()
        else:
            adv = attack(model, x, y, eps)
            pred = model(adv).argmax(1)
            correct += (pred != y).sum().item()

        total += x.size(0)

    return correct / total


# RUN
print("\n=== CLEAN ACCURACY ===")
print("MNIST:", evaluate_model(mnist_model, mnist_testloader))
print("CIFAR:", evaluate_model(cifar_model, cifar_testloader))


# eps = 0.1

# print("\n=== ATTACK (MNIST) ===")
# print("FGSM Targeted:", evaluate_attack(mnist_model, mnist_testloader, fgsm_targeted, True, eps))
# print("FGSM Untargeted:", evaluate_attack(mnist_model, mnist_testloader, fgsm_untargeted, False, eps))
# print("PGD Targeted:", evaluate_attack(
#     mnist_model, mnist_testloader,
#     lambda m,x,y,e: pgd_targeted(m,x,y,10,e,0.01), True, eps
# ))


# print("\n=== ATTACK (CIFAR) ===")
# print("FGSM Targeted:", evaluate_attack(cifar_model, cifar_testloader, fgsm_targeted, True, eps))
# print("FGSM Untargeted:", evaluate_attack(cifar_model, cifar_testloader, fgsm_untargeted, False, eps))
# print("PGD Targeted:", evaluate_attack(
#     cifar_model, cifar_testloader,
#     lambda m,x,y,e: pgd_targeted(m,x,y,10,e,0.01), True, eps
# ))