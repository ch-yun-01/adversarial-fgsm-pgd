import torch
import torch.nn as nn
import torch.nn.functional as F

def margin_loss(logits, label):
    target_logit = logits.gather(1, label.unsqueeze(1)).squeeze()

    mask = torch.ones_like(logits)
    mask.scatter_(1, label.unsqueeze(1), 0)

    other_logits = logits * mask - (1 - mask) * 1e9
    max_other = other_logits.max(1)[0]

    return -(target_logit - max_other).mean()


def _compute_loss(criterion, logits, label, targeted):
    if criterion == 'ce':
        loss = F.cross_entropy(logits, label)

    elif criterion == 'mse':
        probs = F.softmax(logits, dim=1)
        onehot = torch.zeros_like(probs)
        onehot.scatter_(1, label.unsqueeze(1), 1.0)
        loss = F.mse_loss(probs, onehot)

    elif criterion == 'kl':
        log_probs = F.log_softmax(logits, dim=1)
        n_classes = logits.shape[1]
        target_dist = torch.full_like(logits, 0.01 / (n_classes - 1))
        target_dist.scatter_(1, label.unsqueeze(1), 0.99)
        loss = F.kl_div(log_probs, target_dist, reduction='batchmean')

    elif criterion == 'margin':
        loss = margin_loss(logits, label)

    else:
        raise ValueError(f"Unknown criterion '{criterion}'")

    return loss if targeted else -loss


def fgsm_targeted(model, x, target, eps, criterion='ce'):
    """
    model   : the neural network
    x       : input image tensor (requires_grad should be set)
    target  : the desired (wrong) class label
    eps     : perturbation magnitude (e.g., 0.1, 0.3)
    return  : adversarial image x_adv
    """
    model.eval()
    # criterion = nn.CrossEntropyLoss()

    # 1. Compute the model’s output (logits) for the input x
    x_adv = x.clone().detach().requires_grad_(True)
    logits = model(x_adv)

    # 2. Compute the loss between the output and the target label (e.g., using cross-entropy)
    # loss = criterion(model(x_adv), target)
    loss = _compute_loss(criterion, logits, target, True)

    # 3. Backpropagate to compute the gradient of the loss w.r.t. the input: ∇𝑥ℒ(𝑓(𝑥), 𝑦target)
    model.zero_grad(set_to_none=True)
    loss.backward()

    # 4. Generate the adversarial image: 𝑥adv = 𝑥 − 𝜀 · sign(∇𝑥ℒ). 
    # Note the minus sign — we want to minimize the loss for the target class, 
    #                       pushing the prediction toward the target.
    x_adv = x - eps * x_adv.grad.sign()
    
    # 5. Clamp the result to the valid image range [0, 1].
    x_adv = torch.clamp(x_adv, 0, 1)

    return x_adv.detach()




def fgsm_untargeted(model, x, label, eps, criterion='ce'):
    """
    model   : the neural network
    x       : input image tensor
    label   : the correct class label
    eps     : perturbation magnitude
    return  : adversarial image x_adv
    """
    model.eval()

    # 1. Compute the model’s output for x.
    x_adv = x.clone().detach().requires_grad_(True)
    logits = model(x_adv)

    # 2. Compute the loss between the output and the correct label.
    loss = _compute_loss(criterion, logits, label, False)

    # 3. Backpropagate to obtain ∇𝑥ℒ(𝑓(𝑥), 𝑦true).
    model.zero_grad(set_to_none=True)
    loss.backward()

    # 4. Generate the adversarial image: 𝑥adv = 𝑥 + 𝜀 · sign(∇𝑥ℒ). 
    # Note the plus sign — we want to maximize the loss for the correct class, 
    #                      pushing the prediction away from the truth.
    x_adv = x_adv - eps * x_adv.grad.sign()

    # 5. Clamp to [0, 1].
    x_adv = torch.clamp(x_adv, 0, 1)

    return x_adv.detach()
    

def pgd_targeted(model, x, target, k, eps, eps_step, criterion='ce'):
    """
    model    : the neural network
    x        : input image tensor
    target   : desired (wrong) class label
    k        : number of iterations (e.g., 10, 40)
    eps      : total perturbation budget
    eps_step : step size per iteration
    return   : adversarial image x_adv
    """
    model.eval()

    # 1. Initialize the adversarial example. A common choice is to start from the clean input: 𝑥(0)adv = 𝑥.
    x_adv = x.clone().detach()
    
    # 2. For each iteration 𝑖 = 1,... , 𝑘:
    for _ in range(k):
        x_adv = x_adv.detach().requires_grad_(True)
        logits = model(x_adv)

        loss = _compute_loss(criterion, logits, target, False)

        model.zero_grad(set_to_none=True)
        loss.backward()

        # (a) Apply one step of FGSM with step size 𝜀step
        x_adv = x_adv - eps_step * x_adv.grad.sign()
        
        # (b) Project back onto the 𝜀-ball around the original input:
        x_adv = x + torch.clamp(x_adv - x, -eps, eps)
        
        # (c) Clamp to the valid image range [0,1].
        x_adv = torch.clamp(x_adv, 0, 1)

    # 3. Return 𝑥(𝑘)
    return x_adv.detach()

def pgd_untargeted(model, x, label, k, eps, eps_step, criterion='ce'):
    model.eval()

    x_adv = x.clone().detach()

    for _ in range(k):
        x_adv = x_adv.detach().requires_grad_(True)
        logits = model(x_adv)

        loss = _compute_loss(criterion, logits, label, False)

        model.zero_grad(set_to_none=True)
        loss.backward()

        x_adv = x_adv - eps_step * x_adv.grad.sign()
        x_adv = x + torch.clamp(x_adv - x, -eps, eps)
        x_adv = torch.clamp(x_adv, 0, 1)

    return x_adv.detach()



