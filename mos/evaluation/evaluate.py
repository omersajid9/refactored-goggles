import torch

def evaluate(model, dataloader, device, criterion):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch_x, batch_y, batch_lengths in dataloader:
            if batch_x.numel() == 0:
                continue

            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            output, _ = model(batch_x, None)

            output = output.view(-1, output.size(2))
            batch_y = batch_y.view(-1)

            loss = criterion(output, batch_y)
            val_loss += loss.item()
    model.train()
    return val_loss / len(dataloader)