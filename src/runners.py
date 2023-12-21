import torch
from sklearn.metrics import f1_score

# Training
def train(dataloader, model, loss_fn, optimizer, device, epoch, writer):
    model.train()
    total_loss = 0
    total_f1 = 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        y = y.unsqueeze(1).type_as(pred)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        pred_bin = torch.round(torch.sigmoid(pred)).view(-1).cpu().detach().numpy()
        y_true = y.view(-1).cpu().detach().numpy()
        total_f1 += f1_score(y_true, pred_bin, average='binary', zero_division=1)

    avg_loss = total_loss / len(dataloader)
    avg_f1 = total_f1 / len(dataloader)
    writer.add_scalar("Loss/train", avg_loss, epoch)
    writer.add_scalar("F1/train", avg_f1, epoch)
    return avg_loss, avg_f1, pred_bin, y_true

# Validation
@torch.no_grad()
def validate(dataloader, model, loss_fn, device, epoch, writer):
    model.eval()
    total_loss = 0
    total_f1 = 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        y = y.unsqueeze(1).type_as(pred)
        loss = loss_fn(pred, y)
        total_loss += loss.item()

        pred_bin = torch.round(torch.sigmoid(pred)).view(-1).cpu().detach().numpy()
        y_true = y.view(-1).cpu().detach().numpy()
        total_f1 += f1_score(y_true, pred_bin, average='binary', zero_division=1)

    avg_loss = total_loss / len(dataloader)
    avg_f1 = total_f1 / len(dataloader)
    writer.add_scalar("Loss/val", avg_loss, epoch)
    writer.add_scalar("F1/val", avg_f1, epoch)
    return avg_loss, avg_f1
