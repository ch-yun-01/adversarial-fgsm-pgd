import torch
import torchvision
import torchvision.transforms as transforms
import json

from models import MNIST_Net, CIFAR_Net
from train import train
from attack import fgsm_targeted, fgsm_untargeted, pgd_targeted, pgd_untargeted

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# DATA
mnist_transform = transforms.ToTensor()
cifar_transform = transforms.ToTensor()

mnist_train = torchvision.datasets.MNIST('./data', train=True, download=True, transform=mnist_transform)
mnist_test = torchvision.datasets.MNIST('./data', train=False, download=True, transform=mnist_transform)

cifar_train = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=cifar_transform)
cifar_test = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=cifar_transform)

mnist_trainloader = torch.utils.data.DataLoader(mnist_train, batch_size=64, shuffle=True)
mnist_testloader = torch.utils.data.DataLoader(mnist_test, batch_size=64, shuffle=False)

cifar_trainloader = torch.utils.data.DataLoader(cifar_train, batch_size=64, shuffle=True)
cifar_testloader = torch.utils.data.DataLoader(cifar_test, batch_size=64, shuffle=False)


# TRAIN
print("\n=== TRAIN (MNIST) ===")
mnist_model = train(MNIST_Net(), mnist_trainloader, device, ckpt_path="./ckpt/mnist.pth")

print("\n=== TRAIN (CIFAR10) ===")
cifar_model = train(CIFAR_Net(), cifar_trainloader, device, epochs=10, ckpt_path="./ckpt/cifar.pth")


# MODEL ACCURACY
def evaluate_model(model, loader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return correct / total


# ATTACK ACCURACY
def evaluate_attack(model, loader, attack, targeted=False, eps=0.1):
    model.eval()
    correct, total = 0, 0

    for i, (x, y) in enumerate(loader):
        if i >= 100:
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


print("\n=== MODEL ACCURACY ===")
print("MNIST:", evaluate_model(mnist_model, mnist_testloader))
print("CIFAR:", evaluate_model(cifar_model, cifar_testloader))


# TEST ATTACK
eps_list = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.05, 0.1, 0.2, 0.3]
attack_acc = []
loss_func_list = ['ce', 'mse', 'kl', 'margin']

for eps in eps_list:
    for loss in loss_func_list:

        print(f"\n=== ATTACK eps={eps}, loss={loss} ===")

        mnist_result = {
            "dataset": "mnist",
            "eps": eps,
            "loss_func": loss,

            "FGSM Targeted": evaluate_attack(
                mnist_model, mnist_testloader,
                lambda m,x,y,e: fgsm_targeted(m,x,y,e,criterion=loss),
                True, eps
            ),

            "FGSM Untargeted": evaluate_attack(
                mnist_model, mnist_testloader,
                lambda m,x,y,e: fgsm_untargeted(m,x,y,e,criterion=loss),
                False, eps
            ),

            "PGD Targeted": evaluate_attack(
                mnist_model, mnist_testloader,
                lambda m,x,y,e: pgd_targeted(m,x,y,10,e,0.01,criterion=loss),
                True, eps
            ),

            "PGD Untargeted": evaluate_attack(
                mnist_model, mnist_testloader,
                lambda m,x,y,e: pgd_untargeted(m,x,y,10,e,0.01,criterion=loss),
                False, eps
            )
        }

        cifar_result = {
            "dataset": "cifar10",
            "eps": eps,
            "loss_func": loss,

            "FGSM Targeted": evaluate_attack(
                cifar_model, cifar_testloader,
                lambda m,x,y,e: fgsm_targeted(m,x,y,e,criterion=loss),
                True, eps
            ),

            "FGSM Untargeted": evaluate_attack(
                cifar_model, cifar_testloader,
                lambda m,x,y,e: fgsm_untargeted(m,x,y,e,criterion=loss),
                False, eps
            ),

            "PGD Targeted": evaluate_attack(
                cifar_model, cifar_testloader,
                lambda m,x,y,e: pgd_targeted(m,x,y,10,e,0.01,criterion=loss),
                True, eps
            ),

            "PGD Untargeted": evaluate_attack(
                cifar_model, cifar_testloader,
                lambda m,x,y,e: pgd_untargeted(m,x,y,10,e,0.01,criterion=loss),
                False, eps
            )
        }

        attack_acc.append(mnist_result)
        attack_acc.append(cifar_result)

        print(mnist_result)
        print(cifar_result)

    with open('./attack_acc.json', "w") as f:
        json.dump(attack_acc, f, indent=2)