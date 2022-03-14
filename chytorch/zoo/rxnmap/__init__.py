# -*- coding: utf-8 -*-
#
#  Copyright 2021, 2022 Ramil Nugmanov <nougmanoff@protonmail.com>
#  This file is part of chytorch.
#
#  chytorch is free software; you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as published by
#  the Free Software Foundation; either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#  GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with this program; if not, see <https://www.gnu.org/licenses/>.
#
from chython import ReactionContainer
from pkg_resources import resource_stream
from pytorch_lightning import LightningModule
from torch import rand
from torch.nn import LazyLinear
from torch.nn.functional import cross_entropy
from torch.optim import AdamW
from torch.utils.data import DataLoader
from ...nn import ReactionEncoder
from ...optim.lr_scheduler import WarmUpCosine
from ...utils.data import ReactionDataset, collate_reactions, chained_collate


class Unpack:
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return ReactionContainer.unpack(self.data[item])


class Model(LightningModule):
    def __init__(self, *, lr_warmup=1e4, lr_period=5e5, lr_max=1e-4, lr_decrease_coef=.01, masking_rate=.15, **kwargs):
        super().__init__()
        self.encoder = ReactionEncoder(**kwargs)

        self.mlma = LazyLinear(118)
        self.mlmn = LazyLinear(self.encoder.molecule_encoder.centrality_encoder.num_embeddings - 2)

        self.lr_warmup = lr_warmup
        self.lr_period = lr_period
        self.lr_max = lr_max
        self.lr_decrease_coef = lr_decrease_coef
        self.masking_rate = masking_rate
        self.save_hyperparameters(kwargs)

    @classmethod
    def pretrained(cls):
        model = cls.load_from_checkpoint(resource_stream(__package__, 'weights.pt'), map_location='cpu')
        model.eval()
        return model

    def forward(self, batch, *, mapping_task=False):
        if mapping_task:
            return self.encoder(*batch, need_embedding=False, need_weights=True)
        return self.encoder(*batch)

    def training_step(self, batch, batch_idx):
        a, n, d, r = batch
        m = r > 1  # atoms only
        ma = a.masked_fill((rand(a.shape, device=a.device) < self.masking_rate) & m, 2)
        mn = n.masked_fill((rand(n.shape, device=n.device) < self.masking_rate) & m, 1)

        x = self.encoder(ma, mn, d, r)[m]  # atoms only embedding
        atoms = self.mlma(x)
        neighbors = self.mlmn(x)

        l1 = cross_entropy(atoms, a[m].long() - 3)
        l2 = cross_entropy(neighbors, n[m].long() - 2)
        self.log('trn_loss_mlm_a', l1.item(), sync_dist=True)
        self.log('trn_loss_mlm_n', l2.item(), sync_dist=True)
        self.log('trn_loss_tot', l1.item() + l2.item(), sync_dist=True)
        return l1 + l2

    def prepare_dataloader(self, reactions, **kwargs):
        """
        Prepare dataloader for training.

        :param reactions: chython packed reactions list.
        """
        du = Unpack(reactions)
        ds = ReactionDataset(du, distance_cutoff=self.encoder.molecule_encoder.spatial_encoder.num_embeddings - 3)
        return DataLoader(ds, collate_fn=collate_reactions, **kwargs)

    def configure_optimizers(self):
        o = AdamW(self.parameters(), lr=self.lr_max)
        s = WarmUpCosine(o, self.lr_decrease_coef, self.lr_warmup, self.lr_period)
        s.logger = self.log
        return [o], [{'scheduler': s, 'interval': 'step'}]


__all__ = ['Model']
