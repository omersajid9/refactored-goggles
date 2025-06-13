import torch

def evaluate(model, dataloader, device, criterion):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch_x, batch_y, attention_mask in dataloader:
            if batch_x.numel() == 0:
                continue

            batch_x, batch_y, attention_mask = batch_x.to(device), batch_y.to(device), attention_mask.to(device)
            output, _ = model(batch_x, attention_mask)

            batch, vocab = output.size()

            output_flat = output.reshape(batch, vocab)
            batch_y_flat = batch_y.reshape(batch)

            loss = criterion(output_flat, batch_y_flat)
            val_loss += loss.item()
    model.train()
    return val_loss / len(dataloader)