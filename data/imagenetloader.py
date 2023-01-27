import os

from torchvision.datasets import ImageNet, CIFAR100
from torchvision.models import resnet50

from vit_pytorch import ViT
from vit_pytorch.extractor import Extractor
from coca_pytorch.coca_pytorch import CoCa

import os
import shutil
import tempfile
from contextlib import contextmanager
from typing import Any, Dict, List, Iterator, Optional, Tuple

import torch

from torchvision.datasets.folder import ImageFolder
from torchvision.datasets.utils import check_integrity, extract_archive, verify_str_arg

from training import GeneralResnetTrainer

if __name__ == '__main__':
    model = resnet50(pretrained=True)
    fc_in_features = model.fc.in_features
    model.fc = torch.nn.Linear(fc_in_features, 100)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    trainer = GeneralResnetTrainer(model, loss_fn, optimizer, device="cuda", reportToWandb=False)




