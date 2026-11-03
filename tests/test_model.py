import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
import torch
from slm.model import Config, LanguageModel
from slm.train import batch


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.config = Config(vocab_size=16, context=8, width=16, heads=2, layers=2, dropout=0)
        self.model = LanguageModel(self.config)

    def test_causal_mask(self):
        x = torch.randint(16, (2, 8))
        changed = x.clone()
        changed[:, 4:] = (changed[:, 4:] + 1) % 16
        a, _ = self.model(x)
        b, _ = self.model(changed)
        torch.testing.assert_close(a[:, :4], b[:, :4])

    def test_shifted_targets(self):
        x, y = batch(torch.arange(30), 4, 8, 'cpu')
        torch.testing.assert_close(x + 1, y)

    def test_learning_and_checkpoint(self):
        x = torch.arange(8).repeat(4, 1)
        y = (x + 1) % 16
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.02)
        initial = self.model(x, y)[1].item()
        for _ in range(40):
            optimizer.zero_grad()
            loss = self.model(x, y)[1]
            loss.backward()
            optimizer.step()
        self.assertLess(self.model(x, y)[1].item(), initial * 0.2)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'model.pt'
            torch.save({'config': asdict(self.config), 'model': self.model.state_dict()}, path)
            saved = torch.load(path, weights_only=True)
            restored = LanguageModel(Config(**saved['config']))
            restored.load_state_dict(saved['model'])
            torch.testing.assert_close(self.model(x)[0], restored(x)[0])
        generated = self.model.generate(x[:1], 12, top_k=1)
        self.assertEqual(generated.shape, (1, 20))
        torch.testing.assert_close(generated[:, :8], x[:1])
        self.assertTrue(self.model.training)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            Config(width=15, heads=2)
        with self.assertRaises(ValueError):
            self.model(torch.zeros(1, 9, dtype=torch.long))
        with self.assertRaises(ValueError):
            self.model.generate(torch.zeros(1, 1, dtype=torch.long), temperature=0)


if __name__ == '__main__':
    unittest.main()
