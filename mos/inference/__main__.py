
import torch
from mos.data.pipeline import load_data_pipeline
from mos.inference.infer import generate_text

def main():

    data = load_data_pipeline()

    weights_dir = "./weights/abc.pth"
    model = torch.load(weights_dir, weights_only=False)
    generated_text = generate_text(model, data['tokenizer'], prompt="A")
    print(f"Generated text: \"{generated_text}\"")



if __name__ == "__main__":
    main()