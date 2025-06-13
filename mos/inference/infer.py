import torch

def generate_text(model, tokenizer, prompt = "", device='cuda', max_length=25, temperature=1.0):
    model.eval()
    print(tokenizer.char_to_idx)
    prompt_encoded = tokenizer.encode(prompt).tolist()
    input_seq = torch.tensor([[tokenizer.special_tokens['<|im_start|>']] + prompt_encoded], device=device)
    hidden = None
    generated_indices = input_seq[0].tolist() + []

    with torch.no_grad():
        for _ in range(max_length):
            output, hidden = model(input_seq, None, hidden)

            output = output / temperature

            probabilities = torch.softmax(output, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1).item()

            if next_token == tokenizer.special_tokens['<|im_end|>']:
                break

            generated_indices.append(next_token)

            input_seq = torch.tensor([generated_indices], device=device)

    generated_text = tokenizer.decode(torch.tensor(generated_indices))
    model.train()
    return generated_text
