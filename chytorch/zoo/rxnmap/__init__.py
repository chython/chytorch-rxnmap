# -*- coding: utf-8 -*-
#
#  Copyright 2021-2023 Ramil Nugmanov <nougmanoff@protonmail.com>
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
from math import inf
from pkg_resources import resource_stream
from torch import load, zeros_like, float as t_float

from chytorch.nn import ReactionEncoder
from chytorch.utils.data import ReactionEncoderDataset, collate_encoded_reactions


class Model(ReactionEncoder):
    def __init__(self):
        super().__init__()
        self.load_state_dict(load(resource_stream(__package__, 'weights.pt')))
        self.eval()

    def forward(self, reaction):
        dev = self.role_encoder.weight.device
        atoms, neighbors, distances, roles = collate_encoded_reactions([ReactionEncoderDataset([reaction])[0]]).to(dev)
        n = atoms.size(1)
        d_mask = zeros_like(roles, dtype=t_float).masked_fill_(roles == 0, -inf).view(-1, 1, 1, n)
        d_mask = d_mask.expand(-1, self.nhead, n, -1).flatten(end_dim=1)

        x = self.molecule_encoder((atoms, neighbors, distances)) * (roles > 1).unsqueeze_(-1)
        x = x + self.role_encoder(roles)

        for lr in self.layers[:-1]:  # noqa
            x, _ = lr(x, d_mask)
        _, a = self.layers[-1](x, d_mask, need_embedding=False, need_weights=True)
        return a[0]


__all__ = ['Model']
