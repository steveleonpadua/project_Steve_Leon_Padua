import torch

def accuracy(pred, label):
    _, out = torch.max(pred, dim=1)
    return torch.tensor(torch.sum(out == label).item()/len(pred))

def validation_step(valid_dl, model, loss_fn):
    for image, label in valid_dl:
        out = model(image)
        loss = loss_fn(out, label)
        acc = accuracy(out, label)
        return {"val_loss": loss, "val_acc": acc}

def fit(model, num_epochs, train_loader, loss_fn, optimizer):
    history = []
    for epoch in range(num_epochs):
        for image, label in train_loader:
            out = model(image)
            loss = loss_fn(out, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        
        print(f"Epoch [{epoch}/{num_epochs}] => loss: {loss}")
        history.append({"loss": loss})
    return history

