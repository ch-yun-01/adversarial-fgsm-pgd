import torch
import torch.nn as nn


def fgsm_targeted(model, x, target, eps):
    """
    model   : the neural network
    x       : input image tensor (requires_grad should be set)
    target  : the desired (wrong) class label
    eps     : perturbation magnitude (e.g., 0.1, 0.3)
    return  : adversarial image x_adv
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    # 1. Compute the model’s output (logits) for the input x
    x_adv = x.clone().detach()
    x_adv.requires_grad_(True)
    
    # 2. Compute the loss between the output and the target label (e.g., using cross-entropy)
    loss = criterion(model(x_adv), target)

    # 3. Backpropagate to compute the gradient of the loss w.r.t. the input: ∇𝑥ℒ(𝑓(𝑥), 𝑦target)
    model.zero_grad()
    loss.backward()

    # 4. Generate the adversarial image: 𝑥adv = 𝑥 − 𝜀 · sign(∇𝑥ℒ). 
    # Note the minus sign — we want to minimize the loss for the target class, 
    #                       pushing the prediction toward the target.
    x_adv = x - eps * x_adv.grad.sign()
    
    # 5. Clamp the result to the valid image range [0, 1].
    x_adv = torch.clamp(x_adv, 0, 1)

    return x_adv.detach()




def fgsm_untargeted(model, x, label, eps):
    """
    model   : the neural network
    x       : input image tensor
    label   : the correct class label
    eps     : perturbation magnitude
    return  : adversarial image x_adv
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    # 1. Compute the model’s output for x.
    x_adv = x.clone().detach()
    x_adv.requires_grad_(True)

    # 2. Compute the loss between the output and the correct label.
    loss = criterion(model(x_adv), label)

    # 3. Backpropagate to obtain ∇𝑥ℒ(𝑓(𝑥), 𝑦true).
    model.zero_grad()
    loss.backward()

    # 4. Generate the adversarial image: 𝑥adv = 𝑥 + 𝜀 · sign(∇𝑥ℒ). 
    # Note the plus sign — we want to maximize the loss for the correct class, 
    #                      pushing the prediction away from the truth.
    x_adv = x_adv + eps * x_adv.grad.sign()

    # 5. Clamp to [0, 1].
    x_adv = torch.clamp(x_adv, 0, 1)

    return x_adv.detach()
    

def pgd_targeted(model, x, target, k, eps, eps_step):
    """
    model
    : the neural network
    x
    target
    k
    eps
    : input image tensor
    : desired (wrong) class label
    : number of iterations (e.g., 10, 40)
    : total perturbation budget
    eps_step : step size per iteration
    return
    : adversarial image x_adv
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    # 1. Initialize the adversarial example. A common choice is to start from the clean input: 𝑥(0)adv = 𝑥.
    x_adv = x.clone().detach()
    
    # 2. For each iteration 𝑖 = 1,... , 𝑘:
    for _ in range(k):
        x_adv = x_adv.detach().requires_grad_(True)

        loss = criterion(model(x_adv), target)

        model.zero_grad()
        loss.backward()

        # (a) Apply one step of FGSM with step size 𝜀step
        x_adv = x_adv - eps_step * x_adv.grad.sign()
        
        # (b) Project back onto the 𝜀-ball around the original input:
        x_adv = x + torch.clamp(x_adv - x, -eps, eps)
        
        # (c) Clamp to the valid image range [0,1].
        x_adv = torch.clamp(x_adv, 0, 1)

    # 3. Return 𝑥(𝑘)
    return x_adv.detach()



