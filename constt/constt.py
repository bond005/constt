import numpy as np

import torch
from transformers import Wav2Vec2ForCTC


class SiameseWav2Vec2(torch.nn.Module):

    def __init__(self, model: Wav2Vec2ForCTC, n_classes: int):
        super(SiameseWav2Vec2, self).__init__()
        self.model = model
        self.n_classes = n_classes
        pass

    def forward(self):
        pass
