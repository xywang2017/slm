"""Train from scratch on UTF-8 text; no pretrained weights or downloads."""
import argparse
from dataclasses import asdict
from pathlib import Path
import torch
from .model import Config, LanguageModel


def batch(data, batch_size, context, device):
    starts = torch.randint(len(data) - context, (batch_size,))
    x = torch.stack([data[i:i+context] for i in starts])
    y = torch.stack([data[i+1:i+context+1] for i in starts])
    return x.to(device), y.to(device)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('data/tiny.txt'))
    p.add_argument('--output', type=Path, default=Path('checkpoints/tiny.pt'))
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--context', type=int, default=64)
    p.add_argument('--width', type=int, default=64)
    p.add_argument('--layers', type=int, default=2)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=0.003)
    p.add_argument('--eval-every', type=int, default=50)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    if min(args.steps, args.batch_size, args.eval_every, args.threads) <= 0 or args.lr <= 0:
        p.error('steps, batch-size, eval-every, threads and lr must be positive')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    raw = torch.tensor(list(args.data.read_bytes()), dtype=torch.long)
    split = int(0.9 * len(raw))
    train, val = raw[:split], raw[split:]
    if min(len(train), len(val)) <= args.context:
        p.error('each 90/10 data split must contain more bytes than context')
    config = Config(context=args.context, width=args.width, layers=args.layers, heads=args.heads)
    model = LanguageModel(config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print(f'parameters={sum(p.numel() for p in model.parameters()):,}; device={args.device}')
    # A dedicated evaluation RNG keeps evaluation from changing training batches.
    eval_rng = torch.Generator().manual_seed(args.seed + 1)
    fixed = {}
    for name, data in [('train', train), ('val', val)]:
        starts = torch.randint(len(data)-args.context, (args.batch_size,), generator=eval_rng)
        fixed[name] = (torch.stack([data[i:i+args.context] for i in starts]).to(args.device),
                       torch.stack([data[i+1:i+args.context+1] for i in starts]).to(args.device))
    for step in range(args.steps + 1):
        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                losses = {name: model(*pair)[1].item() for name, pair in fixed.items()}
            print(f'step={step} train_loss={losses["train"]:.4f} val_loss={losses["val"]:.4f}', flush=True)
            model.train()
        if step == args.steps:
            break
        x, y = batch(train, args.batch_size, args.context, args.device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'config': asdict(config), 'model': model.state_dict(),
                'steps': args.steps, 'tokenizer': 'utf8-bytes'}, args.output)
    print(f'saved {args.output}')


if __name__ == '__main__':
    main()
