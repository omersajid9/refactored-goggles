import torch
import torch.nn as nn
import torch.optim as optim
import math


def analyze_model_complexity(model):
    """Analyze model complexity and parameter statistics"""
    total_params = 0
    trainable_params = 0
    
    print("=" * 60)
    print("MODEL COMPLEXITY ANALYSIS")
    print("=" * 60)
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        if param.requires_grad:
            trainable_params += param_count
            
        # Parameter statistics
        param_data = param.data.cpu().numpy()
        param_mean = np.mean(param_data)
        param_std = np.std(param_data)
        param_min = np.min(param_data)
        param_max = np.max(param_data)
        
        print(f"{name:25s} | Shape: {str(param.shape):20s} | "
              f"Params: {param_count:8d} | Mean: {param_mean:8.4f} | "
              f"Std: {param_std:7.4f} | Range: [{param_min:7.4f}, {param_max:7.4f}]")
    
    print("-" * 60)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size (MB): {total_params * 4 / (1024**2):.2f}")  # Assuming float32
    print("=" * 60)

def monitor_gradients(model):
    """Monitor gradient statistics during training"""
    total_norm = 0
    param_count = 0
    gradient_stats = {}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
            
            # Detailed gradient statistics
            grad_data = param.grad.data.cpu().numpy()
            gradient_stats[name] = {
                'norm': param_norm.item(),
                'mean': np.mean(grad_data),
                'std': np.std(grad_data),
                'min': np.min(grad_data),
                'max': np.max(grad_data),
                'zero_fraction': np.mean(grad_data == 0)
            }
    
    total_norm = total_norm ** (1. / 2)
    return total_norm, gradient_stats

def log_gradient_stats(gradient_stats, step, log_every=50):
    """Log detailed gradient statistics"""
    if step % log_every == 0:
        print(f"\n--- Gradient Analysis (Step {step}) ---")
        for name, stats in gradient_stats.items():
            print(f"{name:25s} | Norm: {stats['norm']:8.4f} | "
                  f"Mean: {stats['mean']:8.4f} | Std: {stats['std']:7.4f} | "
                  f"Zero%: {stats['zero_fraction']*100:5.1f}")
        print("-" * 60)

def detect_gradient_anomalies(gradient_stats, total_norm, step):
    """Detect potential gradient problems"""
    warnings = []
    
    # Check for gradient explosion
    if total_norm > 10.0:
        warnings.append(f"⚠️  Large gradient norm: {total_norm:.4f}")
    
    # Check for vanishing gradients
    if total_norm < 1e-6:
        warnings.append(f"⚠️  Very small gradient norm: {total_norm:.6f}")
    
    # Check individual layer gradients
    for name, stats in gradient_stats.items():
        if stats['norm'] > 5.0:
            warnings.append(f"⚠️  Large gradient in {name}: {stats['norm']:.4f}")
        if stats['zero_fraction'] > 0.9:
            warnings.append(f"⚠️  Many zero gradients in {name}: {stats['zero_fraction']*100:.1f}%")
    
    if warnings:
        print(f"\n🚨 GRADIENT WARNINGS (Step {step}):")
        for warning in warnings:
            print(f"   {warning}")
        print()


def generate_text(model, encoder, device='cuda', max_length=100, temperature=1.0):
    model.eval()
    input_seq = torch.tensor([[encoder.start_idx]], device=device)
    hidden = None
    generated_indices = []

    with torch.no_grad():
        for _ in range(max_length):
            output, hidden = model(input_seq, hidden)
            output = output[:, -1, :]  # get the last time step

            output = output / temperature

            probabilities = torch.softmax(output, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1).item()

            if next_token == encoder.eos_idx:
                break

            generated_indices.append(next_token)

            input_seq = torch.tensor([[next_token]], device=device)

    generated_text = encoder.decode(torch.tensor(generated_indices))
    return generated_text

class TrainPipeline:
    @staticmethod
    def train_loop(model, encoder, dataloader, val_dataloader, epochs=100, device='cuda'):
        model.to(device)
        criterion = nn.CrossEntropyLoss(ignore_index=encoder.pad_idx)
        optimizer = optim.Adam(model.parameters(), lr=0.0005) #use different optimizer

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )

        generated_text = generate_text(model, encoder)
        print(f"Generated text before training \"{generated_text}\"")


        global_step = 0
        for epoch in range(epochs):
            epoch_loss = 0.0

            for batch_idx, (batch_x, batch_y, batch_length) in enumerate(dataloader):
                model.train()
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                if batch_x.numel() == 0:
                    continue

                scheduler.zero_grad()
                output, _ = model(batch_x, None)


                output = output.view(-1, output.size(2))
                batch_y = batch_y.view(-1)

                loss = criterion(output, batch_y)

                loss.backward()

                # Monitor gradients BEFORE clipping
                total_grad_norm, gradient_stats = monitor_gradients(model)

                # gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Monitor gradients AFTER clipping
                clipped_grad_norm, _ = monitor_gradients(model)


                scheduler.step()

                epoch_loss += loss.item()
                global_step += 1

                if global_step % 1 == 0:
                    avg_loss = epoch_loss / (batch_idx + 1)

                    val_loss = TrainPipeline.evaluate(model, val_dataloader, device, criterion)

                    train_perplexity = math.exp(avg_loss)
                    val_perplexity = math.exp(val_loss)


                    # Basic training stats
                    print(f'Epoch {epoch}, Step {global_step}, Train Loss: {avg_loss:.4f}, '
                          f'Train Perplexity: {train_perplexity:.4f}, Val Loss: {val_loss:.4f}, '
                          f'Val Perplexity: {val_perplexity:.4f}')

                    # Gradient information
                    was_clipped = "✂️" if total_grad_norm > max_grad_norm else "  "
                    print(f'   Grad Norm: {total_grad_norm:.4f} → {clipped_grad_norm:.4f} {was_clipped} | '
                          f'LR: {optimizer.param_groups[0]["lr"]:.6f}')


                    # Detect gradient anomalies
                    detect_gradient_anomalies(gradient_stats, total_grad_norm, global_step)
                    
                    # Detailed gradient logging (less frequent)
                    log_gradient_stats(gradient_stats, global_step, log_every=25)
                    
                    # Learning rate scheduling
                    scheduler.step(val_loss)

                    print(f'Epoch {epoch}, Step {global_step}, Train Loss: {avg_loss:.4f}, Train Perplexity: {train_perplexity:.4f}, Val Loss: {val_loss:.4f}, Val Perplexity: {val_perplexity:.4f}')

                if global_step % 100 == 0:
                    generated_text = generate_text(model, encoder)
                    print(f"Generated text: \"{generated_text}\"")


    @staticmethod
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

        return val_loss / len(dataloader)


class Seq2SeqModel(nn.Module):
    def __init__(self, vocab_size, hidden_size = 256, num_layers = 2, dropout = 0.5):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.lstm_dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        embedd = self.embedding_layer(x)
        output, hidden = self.lstm(embedd, hidden)
        output = self.lstm_dropout(output)
        y_hat = self.linear2(self.relu(self.linear1(output)))
        # y_hat = self.softmax(y_hat)
        return y_hat, hidden

    def save(self):
        pass



    