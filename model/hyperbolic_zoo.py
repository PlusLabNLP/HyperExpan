# Adapted from: https://github.com/HazyResearch/hgcn/blob/master/models

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import hype.hgcn_manifolds as manifolds
from hype.layers.att_layers import GraphAttentionLayer
import hype.layers.hyp_layers as hyp_layers
from hype.layers.layers import GraphConvolution, Linear, get_dim_act, FermiDiracDecoder

class Encoder(nn.Module):
    """
    Encoder abstract class.
    """

    def __init__(self, c):
        super(Encoder, self).__init__()
        self.c = c

    def encode(self, x, adj):
        if self.encode_graph:
            input = (x, adj)
            output, _ = self.layers.forward(input)
        else:
            output = self.layers.forward(x)
        return output

class HNN(Encoder):
    """
    Hyperbolic Neural Networks.
    """

    def __init__(self, c, args):
        super(HNN, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        assert args.num_layers > 1

        # if use trainable curvature
        if args.c is None:
            c_term = nn.Parameter(torch.Tensor([1.]))
        else:
            c_term = None
        self.c = c_term

        dims, acts, self.curvatures = hyp_layers.get_dim_act_curv(args)
        hnn_layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            hnn_layers.append(
                    hyp_layers.HNNLayer(
                            self.manifold, in_dim, out_dim, self.c, args.dropout, act, args.bias)
            )
        self.layers = nn.Sequential(*hnn_layers)
        # self.layers = nn.Sequential(
        #     hyp_layers.HNNLayer(self.manifold, 250, 250, self.c, args.dropout, act, args.bias),
        #     hyp_layers.HNNLayer(self.manifold, 250, args.dim, self.c, args.dropout, act, False)
        # )
        # self.layers = nn.Sequential(
        #     hyp_layers.HNNLayer(self.manifold, 300, 100, self.c, args.dropout, act, args.bias)
        # )
        self.encode_graph = False

    def forward(self, x, adj=None):
        # x in Euclidean space
        x_hyp = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(x, self.c), c=self.c), c=self.c)
        # forward
        output_hyp = self.layers.forward(x)
        # map to Euclidean space
        output_euc = self.manifold.logmap0(output_hyp, c=self.c)
        return output_euc


class HGCN(Encoder):
    """
    Hyperbolic-GCN.
    """
    def __init__(self, c, args):
        super(HGCN, self).__init__(c)
        self.args = args
        self.args.one_rel_pos_emb = True
        self.manifold = getattr(manifolds, args.manifold)()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        assert args.num_layers > 1
        rel_position_vocab_size = 6

        # if use trainable curvature
        if args.c is None:
            c_term = nn.Parameter(torch.Tensor([1.]))
        else:
            c_term = None

        dims, acts, self.curvatures = hyp_layers.get_dim_act_curv(args, curv_param=c_term)
        self.dims = dims
        if args.c is None:
            self.curvatures.append(c_term)
        else:
            self.curvatures.append(self.c)
        hgc_layers = []
        self.concat_funcs = nn.ModuleList()
        if self.args.rel_pos_emb_dim > 0:
            self.rel_pos_embs = nn.ModuleList()
            if self.args.one_rel_pos_emb:
                # use one relation position embedding for all layers
                self.rel_pos_embs.append(nn.Embedding(rel_position_vocab_size, self.args.rel_pos_emb_dim))
        else:
            self.args.rel_pos_emb_dim = 0
        if self.args.abs_pos_emb_dim > 0:
            self.abs_pos_embs = nn.ModuleList()
            self.abs_pos_embs.append(nn.Embedding(self.args.max_depth, self.args.abs_pos_emb_dim))
        else:
            self.args.abs_pos_emb_dim = 0
        
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i + 1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            hgc_layers.append(
                    hyp_layers.HyperbolicGraphConvolution(
                            self.manifold, in_dim, out_dim, c_in, c_out,
                            args.dropout, act, args.bias, args.use_att,
                            args.local_agg
                    )
            )
            if self.args.rel_pos_emb_dim > 0:
                if not self.args.one_rel_pos_emb:
                    self.rel_pos_embs.append(nn.Embedding(rel_position_vocab_size, self.args.rel_pos_emb_dim))
            if self.args.rel_pos_emb_dim + self.args.abs_pos_emb_dim > 0:
                concat_func_this = hyp_layers.HyperbolicConcat(self.manifold, 
                            self.dims[i], self.dims[i], max(self.args.rel_pos_emb_dim, self.args.abs_pos_emb_dim), 
                            c_term, args.dropout, dim_c=self.args.abs_pos_emb_dim).to(self.device)
                self.concat_funcs.append(concat_func_this)
        self.layers = nn.Sequential(*hgc_layers)
        self.encode_graph = True

    def encode(self, x, adj, g):
        if self.args.transform_mode == 1:
            # input x is Euclidean, map to hyperbolic
            x_tan = self.manifold.proj_tan0(x, self.curvatures[0])
            x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])
            x_hyp = self.manifold.proj(x_hyp, c=self.curvatures[0])
        elif self.args.transform_mode == 2:
            # input x is hyperbolic, no need to map to hyperbolic
            x_hyp = x
            x_hyp = self.manifold.proj(x_hyp, c=self.curvatures[0])
        elif self.args.transform_mode == 3:
            # input x is hyperbolic, map to Euclidean
            x_hyp = self.manifold.proj_tan0(self.manifold.logmap0(x, c=self.curvatures[0]), c=self.curvatures[0])
        
        if self.args.rel_pos_emb_dim <= 0 and self.args.abs_pos_emb_dim <= 0:
            return super(HGCN, self).encode(x_hyp, adj)
        else:
            # use separate encode part
            positions = g.ndata.pop('pos').to(x_hyp.device)
            positions_abs = g.ndata.pop('abs_level').to(x_hyp.device)
            for idx, layer in enumerate(self.layers):
                # relative position embedding
                if self.args.rel_pos_emb_dim > 0:
                    if self.args.one_rel_pos_emb:
                        p = self.rel_pos_embs[0](positions)
                    else:
                        p = self.rel_pos_embs[idx](positions)
                    p_tan = self.manifold.proj_tan0(p, self.curvatures[idx])
                    p_hyp = self.manifold.expmap0(p_tan, c=self.curvatures[idx])
                    p_hyp = self.manifold.proj(p_hyp, c=self.curvatures[idx])

                # absolute position embedding
                if self.args.abs_pos_emb_dim > 0:
                    p_abs = self.abs_pos_embs[0](positions_abs)
                    p_abs_tan = self.manifold.proj_tan0(p_abs, self.curvatures[idx])
                    p_abs_hyp = self.manifold.expmap0(p_abs_tan, c=self.curvatures[idx])
                    p_abs_hyp = self.manifold.proj(p_abs_hyp, c=self.curvatures[idx])

                # load hyperbolic concat class
                if self.args.rel_pos_emb_dim > 0 and self.args.abs_pos_emb_dim > 0:
                    # concatenate both position embedding
                    concat_func = self.concat_funcs[idx]
                    new_input = concat_func(x_hyp, p_hyp, input_c=p_abs_hyp)
                elif self.args.rel_pos_emb_dim > 0:
                    # only concat rel pos
                    concat_func = self.concat_funcs[idx]
                    new_input = concat_func(x_hyp, p_hyp)
                elif self.args.abs_pos_emb_dim > 0:
                    # only concat abs pos
                    concat_func = self.concat_funcs[idx]
                    new_input = concat_func(x_hyp, p_abs_hyp)

                if self.encode_graph:
                    x_hyp, _ = layer((new_input, adj))
                else:
                    x_hyp = layer(new_input)
            return x_hyp

    def forward(self, g, features):
        adj = g.adjacency_matrix()
        adj = adj.to(self.args.device)
        output_hyp = self.encode(features, adj, g) # output is hyperbolic
        # convert to Euclidean
        output_euc = self.manifold.proj_tan0(self.manifold.logmap0(output_hyp, c=self.curvatures[-1]), c=self.curvatures[-1])
        return output_euc


class Decoder(nn.Module):
    """
    Decoder abstract class for node classification tasks.
    """

    def __init__(self, c):
        super(Decoder, self).__init__()
        self.c = c

    def decode(self, x, adj):
        if self.decode_adj:
            input = (x, adj)
            probs, _ = self.cls.forward(input)
        else:
            probs = self.cls.forward(x)
        return probs

class LinearDecoder(Decoder):
    """
    MLP Decoder for Hyperbolic/Euclidean node classification models.
    """

    def __init__(self, c, args, l_dim, r_dim):
        super(LinearDecoder, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        self.input_dim = l_dim + r_dim
        self.output_dim = 1
        self.bias = args.bias
        self.cls = Linear(self.input_dim, self.output_dim, args.dropout, lambda x: x, self.bias)
        self.decode_adj = False

    def forward(self, x, q, adj=None):
        h1 = self.manifold.proj_tan0(self.manifold.logmap0(x, c=self.c), c=self.c)
        h2 = self.manifold.proj_tan0(self.manifold.logmap0(q, c=self.c), c=self.c)
        concat_h = torch.cat((h1, h2), 1)
        return super(LinearDecoder, self).decode(concat_h, adj=adj)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, c={}'.format(
                self.input_dim, self.output_dim, self.bias, self.c
        )

class FermiDiracDecoderContainer(Decoder):
    def __init__(self, c, args, l_dim, r_dim):
        super(FermiDiracDecoderContainer, self).__init__(c)
        self.dc = FermiDiracDecoder(r=args.r, t=args.t)
        self.manifold_name = args.manifold
        self.manifold = getattr(manifolds, args.manifold)()
        self.decode_adj = False
        self.c = c

    def forward(self, x, q, adj=None):
        # convert to hyperbolic space
        x_hyp = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(x, self.c), c=self.c), c=self.c)
        q_hyp = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(q, self.c), c=self.c), c=self.c)
        sqdist = self.manifold.sqdist(x_hyp, q_hyp, self.c)
        probs = self.dc.forward(sqdist)
        return probs