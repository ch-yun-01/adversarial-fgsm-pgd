import os
import torch
import torch.nn.functional as F
import torch.optim as optim

def train(model, loader, device, epochs=5, ckpt_path=None):
    model.to(device)

    # LOAD
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"[INFO] Load checkpoint: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        return model

    print("[INFO] No checkpoint → Training")

    optimizer = optim.Adam(model.parameters(), lr=0.001)


    # TRAIN
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for x,y in loader:
            x,y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out,y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.3f}")

    # SAVE
    if ckpt_path:
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)
        print(f"[INFO] Saved checkpoint: {ckpt_path}")

    return model