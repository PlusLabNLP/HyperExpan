#!/usr/bin/env python3
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
import torch as th
import geoopt as gt
import torch.nn as nn
import geoopt.manifolds.stereographic.math as pmath
from .hypernn import *


class EnergyFunction(torch.nn.Module):
    def __init__(self, manifold, dim, size, objects, sparse=False, evalonly=False, **kwargs):
        super().__init__()
        self.manifold = manifold
        self.lt = manifold.allocate_lt(size, dim, sparse)
        self.nobjects = size
        self.num_rels = 1 # number of relations used to init relation embeddings
        self.design = kwargs['design']
        self.design_config = kwargs['design_config']
        self.output_dim = dim

        self.manifold.init_weights(self.lt, objects, evalonly=evalonly, 
                                    load_vec=kwargs['pretrained_embedding'], 
                                    vec_choice=self.design_config['emb_vec_choice'])
        self.c = 1

        # transform
        # 1: just a weight matrix
        # 2: just bias term
        # 3: both weight matrix and bias term
        # 4: multiple layers
        # 5: Euclidean exp map to Poincare ball

        if self.design == 'transformation':
            self.input_dropout = nn.Dropout(p=0.3)

            layers = []
            for i in range(self.design_config['num_linear_layers']):
                layers.append(MobiusLinear(dim, self.output_dim, bias=True, nonlin=nn.Tanh(), c=self.c))

            self.lins = nn.Sequential(*layers)

            self.ball = gt.PoincareBall(c=self.c)

            if self.design_config['rel_emb'] == True:
                # initialize relation embeddings
                # r_emb = torch.empty(self.output_dim).uniform_(-0.001, 0.001)
                r_emb = torch.zeros(self.output_dim)
                self.r_emb = gt.ManifoldParameter(pmath.expmap0(r_emb, k=self.ball.k), manifold=self.ball)

        if not kwargs['embedding_freeze']:
            # Option 1: update embedding layer
            self.params_list = list(self.lt.parameters())
        else:
            # Option 2: do not update embedding layer
            self.params_list = []
            for param in self.lt.parameters():
                param.requires_grad = False

    def transform(self, e, rel_emb = False):
        if self.design_config['emb_vec_choice'] != 'poincareglove':
            # map Euc features to poincare ball
            e = pmath.expmap0(e, k=self.ball.k)
        e = self.lins(e)

        if self.design_config['rel_emb'] == True and rel_emb:
            # e.size()[0]
            e = pmath.mobius_add(e, self.r_emb, k=self.ball.k)

        return e

    def forward(self, inputs):
        # e is the embedding matrix OR
        # e is the embedding matrix but initialized by a pre-trained vectors by different init_weights function
        e = self.lt(inputs)
        
        # dropout
        with torch.no_grad():
            e = self.manifold.normalize(e)

        if self.design == 'transformation':
            e = self.transform(e)

        o = e.narrow(1, 1, e.size(1) - 1)
        s = e.narrow(1, 0, 1).expand_as(o)

        if self.design == 'transformation':
            if self.design_config['rel_emb'] == True:
                s = pmath.mobius_add(s, self.r_emb, k=self.ball.k)
        
        output = self.energy(s, o).squeeze(-1)
        return output

    def optim_params(self):
        # This is discarded if use geoopt optimizer
        # Note: need to distinguish different manifold for params in different geometry
        return [{
            'params': self.params_list,
            'rgrad': self.manifold.rgrad,
            'expm': self.manifold.expm,
            'logm': self.manifold.logm,
            'ptransp': self.manifold.ptransp,
        }]

    def loss_function(self, inp, target, **kwargs):
        raise NotImplementedError


class DistanceEnergyFunction(EnergyFunction):
    def energy(self, s, o):
        return self.manifold.distance(s, o)

    def loss(self, inp, target, **kwargs):
        return F.cross_entropy(inp.neg(), target)


class EntailmentConeEnergyFunction(EnergyFunction):
    def __init__(self, *args, margin=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.manifold.K is not None, (
            "K cannot be none for EntailmentConeEnergyFunction"
        )
        assert hasattr(self.manifold, 'angle_at_u'), 'Missing `angle_at_u` method'
        self.margin = margin

    def energy(self, s, o):
        energy = self.manifold.angle_at_u(o, s) - self.manifold.half_aperture(o)
        return energy.clamp(min=0)

    def loss(self, inp, target, **kwargs):
        loss = inp[:, 0].clamp_(min=0).sum()  # positive
        loss += (self.margin - inp[:, 1:]).clamp_(min=0).sum()  # negative
        return loss / inp.numel()
