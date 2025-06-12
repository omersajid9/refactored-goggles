import torch

def train_loop(model, encoder, dataloader, val_dataloader, epochs=50, device='cuda'):
    torch.cuda.empty_cache()
    model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=encoder.pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    print(f"Train dataset \"{len(dataloader)}\"")
    print(f"Val dataset \"{len(val_dataloader)}\"")

    generated_text = generate_text(model, encoder)
    print(f"Generated text before training \"{generated_text}\"")

    max_grad_norm = 1.0
    noticable = 100000

    global_step = 0
    for epoch in range(epochs):
        epoch_loss = 0.0

        for batch_idx, (batch_x, batch_y, batch_length) in enumerate(dataloader):
            model.train()
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)


            # print(encoder.decode(batch_x[0]))
            # print(encoder.decode(batch_y[0]))
            # break
            if batch_x.numel() == 0:
                continue

            optimizer.zero_grad()
            output, _ = model(batch_x, None)


            output = output.view(-1, output.size(2))
            batch_y = batch_y.view(-1)

            loss = criterion(output, batch_y)

            loss.backward()

            # Monitor gradients BEFORE clipping
            total_grad_norm, gradient_stats = monitor_gradients(model)

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

            # Monitor gradients AFTER clipping
            clipped_grad_norm, _ = monitor_gradients(model)


            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % 1000 == 0:
                avg_loss = epoch_loss / (batch_idx + 1)

                val_loss = TrainPipeline.evaluate(model, val_dataloader, device, criterion)

                train_perplexity = math.exp(avg_loss)
                val_perplexity = math.exp(val_loss)


                # Basic training stats
                print(f'Epoch {epoch}, Step {global_step}, Train Loss: {avg_loss:.4f}, '
                        f'Train Perplexity: {train_perplexity:.4f}, Val Loss: {val_loss:.4f}, '
                        f'Val Perplexity: {val_perplexity:.4f}')

                # Gradient information
                was_clipped = "✂️" if int(noticable * total_grad_norm) > int(noticable * max_grad_norm) else "  "
                print(f'   Grad Norm: {total_grad_norm:.4f} → {clipped_grad_norm:.4f} {was_clipped} | '
                        f'LR: {optimizer.param_groups[0]["lr"]:.6f}')


                # Detect gradient anomalies
                detect_gradient_anomalies(gradient_stats, total_grad_norm, global_step)
                
                # Detailed gradient logging (less frequent)
                log_gradient_stats(gradient_stats, global_step, log_every=10000)
                
                # Learning rate scheduling
                scheduler.step(val_loss)

            if int(optimizer.param_groups[0]["lr"] * noticable) == 0:
                break
            if global_step % 1000 == 0:
                generated_text = generate_text(model, encoder)
                print(f"Generated text: \"{generated_text}\"")
                generated_text = generate_text(model, encoder)
                print(f"Generated text: \"{generated_text}\"")

