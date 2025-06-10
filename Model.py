import torch
import torch.nn as nn
import torch.optim as optim


def generate_name(model, encoder, device='mps', max_length=20, temperature=1.0):
    model.eval()
    input_seq = torch.tensor([[encoder.char_to_idx.get('<EOS>', 0)]], device=device)
    hidden = None
    generated_indices = []

    with torch.no_grad():
        for _ in range(max_length):
            output, hidden = model(input_seq, hidden)
            output = output[:, -1, :]  # get the last time step

            output = output / temperature

            probabilities = torch.softmax(output, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1).item()

            if encoder.idx_to_char[next_token] == '<EOS>':
                break

            generated_indices.append(next_token)

            input_seq = torch.tensor([[next_token]], device=device)

    generated_name = encoder.decode(torch.tensor(generated_indices))
    return generated_name

class TrainPipeline:
    @staticmethod
    def train_loop(model, encoder, dataloader, val_dataloader, epochs=1000, device='mps'):
        model.to(device)
        criterion = nn.CrossEntropyLoss(ignore_index=encoder.pad_idx)
        optimizer = optim.Adam(model.parameters(), lr=0.001) #use different optimizer

        global_step = 0
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0

            for batch_idx, (batch_x, batch_y, batch_length) in enumerate(dataloader):
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                if batch_x.numel() == 0:
                    continue

                optimizer.zero_grad()
                output, _ = model(batch_x, None)


                output = output.view(-1, output.size(2))
                batch_y = batch_y.view(-1)

                loss = criterion(output, batch_y)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                global_step += 1

                if global_step % 100 == 0:
                    avg_loss = epoch_loss / (batch_idx + 1)

                    val_loss = TrainPipeline.evaluate(model, val_dataloader, device, criterion)
                    print(f'Epoch {epoch}, Step {global_step}, Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}')

                    generated_name = generate_name(model, encoder)
                    print(f"Generated name \"{generated_name}\"")


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
    def __init__(self, vocab_size, hidden_size = 512, num_layers = 2, dropout = 0.5):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        embedd = self.embedding_layer(x)
        output, hidden = self.lstm(embedd, hidden)
        drop_output = self.dropout(output)
        y_hat = self.linear(drop_output)
        # y_hat = self.softmax(y_hat)
        return y_hat, hidden


    