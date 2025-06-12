import torch

def generate_text(model, tokenizer, prompt = "", device='cuda', max_length=25, temperature=1.0):
    model.eval()
    prompt_encoded = tokenizer.encode(prompt)
    input_seq = torch.tensor([[tokenizer.special_tokens['<|im_start|>'] + prompt_encoded]], device=device)
    hidden = None
    generated_indices = []

    with torch.no_grad():
        for _ in range(max_length):
            output, hidden = model(input_seq, hidden)
            output = output[:, -1, :]  # get the last time step

            output = output / temperature

            probabilities = torch.softmax(output, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1).item()

            if next_token == tokenizer.eos_idx:
                break

            generated_indices.append(next_token)

            input_seq = torch.tensor([[next_token]], device=device)

    generated_text = tokenizer.decode(torch.tensor(generated_indices))
    return generated_text
