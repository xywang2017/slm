"""Load a trusted local checkpoint and sample one byte at a time."""
import argparse
import torch
from .model import Config, LanguageModel


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', default='checkpoints/tiny.pt')
    p.add_argument('--prompt', default='The little robot')
    p.add_argument('--tokens', type=int, default=200)
    p.add_argument('--temperature', type=float, default=0.8)
    p.add_argument('--top-k', type=int, default=40)
    p.add_argument('--device', default='cpu')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if not args.prompt:
        p.error('prompt must not be empty')
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    model = LanguageModel(Config(**checkpoint['config'])).to(args.device)
    model.load_state_dict(checkpoint['model'])
    ids = torch.tensor([list(args.prompt.encode('utf-8'))], device=args.device)
    result = model.generate(ids, args.tokens, args.temperature, args.top_k)
    print(bytes(result[0].tolist()).decode('utf-8', errors='replace'))


if __name__ == '__main__':
    main()
