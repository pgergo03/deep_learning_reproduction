from reproducibility import make_generator, set_seed

import random
import unittest

import numpy as np
import torch
import torchattacks
import kornia.augmentation as ka
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from my_utils import Bil_layer


class ReproducibilityTests(unittest.TestCase):
    seed = 41

    def test_python_numpy_and_torch_rngs_reset(self):
        def sample():
            set_seed(self.seed)
            return random.random(), np.random.random(), torch.rand(4)

        first = sample()
        second = sample()
        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])
        self.assertTrue(torch.equal(first[2], second[2]))

    def test_dataloader_shuffle_repeats(self):
        dataset = TensorDataset(torch.arange(20))

        def order():
            loader = DataLoader(
                dataset,
                batch_size=5,
                shuffle=True,
                generator=make_generator(self.seed),
            )
            return torch.cat([batch[0] for batch in loader])

        self.assertTrue(torch.equal(order(), order()))

    def test_bilateral_layer_noise_repeats(self):
        layer = Bil_layer(bil=0, noise=0.2, device="cpu")
        images = torch.zeros(2, 3, 4, 4)

        set_seed(self.seed)
        first = layer(images)
        set_seed(self.seed)
        second = layer(images)
        self.assertTrue(torch.equal(first, second))

    def test_kornia_noise_repeats(self):
        augmentation = ka.RandomGaussianNoise(mean=0, std=0.2, p=1)
        images = torch.zeros(2, 3, 4, 4)

        set_seed(self.seed)
        first = augmentation(images)
        set_seed(self.seed)
        second = augmentation(images)
        self.assertTrue(torch.equal(first, second))

    def test_training_updates_repeat(self):
        features = torch.arange(24, dtype=torch.float32).reshape(12, 2) / 24
        labels = torch.arange(12) % 2
        dataset = TensorDataset(features, labels)

        def train_once():
            set_seed(self.seed)
            model = nn.Linear(2, 2)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            loader = DataLoader(
                dataset,
                batch_size=3,
                shuffle=True,
                generator=make_generator(self.seed),
            )
            for inputs, targets in loader:
                inputs = inputs + torch.randn_like(inputs) * 0.01
                loss = nn.functional.cross_entropy(model(inputs), targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            return model.state_dict()

        first = train_once()
        second = train_once()
        for name in first:
            self.assertTrue(torch.equal(first[name], second[name]))

    def test_random_start_attacks_repeat(self):
        images = torch.tensor(
            [[[[0.2, 0.4], [0.6, 0.8]]], [[[0.8, 0.6], [0.4, 0.2]]]]
        )
        labels = torch.tensor([0, 1])

        def run(attack_type):
            set_seed(self.seed)
            model = nn.Sequential(nn.Flatten(), nn.Linear(4, 2))
            if attack_type is torchattacks.APGD:
                attack = attack_type(model, eps=0.25, steps=3, seed=self.seed)
            else:
                attack = attack_type(model, eps=0.25, steps=3, eot_iter=2)
            return attack(images, labels)

        for attack_type in (torchattacks.APGD, torchattacks.EOTPGD):
            self.assertTrue(torch.equal(run(attack_type), run(attack_type)))


if __name__ == "__main__":
    unittest.main()
