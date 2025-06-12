import math
import torch
import torch.nn as nn
import torch.optim as optim

from mos.evaluation.evaluate import evaluate
from mos.utils.metrics import detect_gradient_anomalies, monitor_gradients

def train_loop(model, data, epochs = 10, device = 'cuda'):
    torch.cuda.empty_cache()

    model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=data['tokenizer'].special_tokens['<|pad|>'])

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    max_grad_norm = 1.0
    noticable = 100000

    model.train()

    global_step = 0
    for epoch in range(epochs):
        epoch_loss = 0.0

        for batch_idx, (batch_x, batch_y, batch_length) in enumerate(data['dataloader']['train']):

            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            if batch_x.numel() == 0:
                continue

            optimizer.zero_grad()
            output, _ = model(batch_x, None)

            batch, seq_len, vocab = output.size()

            output_flat = output.reshape(batch * seq_len, vocab)
            batch_y_flat = batch_y.reshape(batch * seq_len)

            loss = criterion(output_flat, batch_y_flat)
            loss.backward()

            total_grad_norm, gradient_stats = monitor_gradients(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            clipped_grad_norm, _ = monitor_gradients(model)

            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % 1000 == 0:
                avg_loss = epoch_loss / (batch_idx + 1)

                val_loss = evaluate(model, data['dataloader']['val'], device, criterion)

                train_perplexity = math.exp(avg_loss)
                val_perplexity = math.exp(val_loss)

                print(f'Epoch {epoch}, Step {global_step}, Train Loss: {avg_loss:.4f}, '
                        f'Train Perplexity: {train_perplexity:.4f}, Val Loss: {val_loss:.4f}, '
                        f'Val Perplexity: {val_perplexity:.4f}')
                
                detect_gradient_anomalies(gradient_stats, total_grad_norm, global_step)
                # log_gradient_stats(gradient_stats, global_step, log_every=10000)
                
                was_clipped = "✂️" if int(noticable * total_grad_norm) > int(noticable * max_grad_norm) else "  "
                print(f'   Grad Norm: {total_grad_norm:.4f} → {clipped_grad_norm:.4f} {was_clipped} | '
                        f'LR: {optimizer.param_groups[0]["lr"]:.6f}')

                scheduler.step(val_loss)

        if int(optimizer.param_groups[0]["lr"] * noticable) == 0:
            break
#             if global_step % 1000 == 0:
#                 generated_text = generate_text(model, encoder)
#                 print(f"Generated text: \"{generated_text}\"")
#                 generated_text = generate_text(model, encoder)
#                 print(f"Generated text: \"{generated_text}\"")
