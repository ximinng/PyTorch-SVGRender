__version__ = '0.3.0'

import torch
from einops import rearrange, repeat

from .inception import InceptionV3
from .fid_score import calculate_frechet_distance


class PytorchFIDFactory(torch.nn.Module):
    """

   Args:
       channels:
       inception_block_idx:

    Examples:
    >>> fid_factory =  PytorchFIDFactory()
    >>> fid_score = fid_factory.score(real_samples=data, fake_samples=all_images)
    >>> print(fid_score)
   """

    def __init__(self, channels: int = 3, inception_block_idx: int = 2048):
        super().__init__()
        self.channels = channels

        # load models
        assert inception_block_idx in InceptionV3.BLOCK_INDEX_BY_DIM
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[inception_block_idx]
        self.inception_v3 = InceptionV3([block_idx])

    @torch.no_grad()
    def calculate_activation_statistics(self, samples):
        features = self.inception_v3(samples)[0]
        features = rearrange(features, '... 1 1 -> ...')

        mu = torch.mean(features, dim=0).cpu()
        sigma = torch.cov(features).cpu()
        return mu, sigma

    def score(self, real_samples, fake_samples):
        if self.channels == 1:
            real_samples, fake_samples = map(
                lambda t: repeat(t, 'b 1 ... -> b c ...', c=3), (real_samples, fake_samples)
            )

        min_batch = min(real_samples.shape[0], fake_samples.shape[0])
        real_samples, fake_samples = map(lambda t: t[:min_batch], (real_samples, fake_samples))

        m1, s1 = self.calculate_activation_statistics(real_samples)
        m2, s2 = self.calculate_activation_statistics(fake_samples)

        fid_value = calculate_frechet_distance(m1, s1, m2, s2)
        return fid_value
