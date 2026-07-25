import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ASRB(nn.Module):
    # Use learnable queries + Transformer as bottleneck to get M prototypes,
    # decoder needs to reconstruct M prototypes into N patches
    def __init__(
            self,
            encoder,
            bottleneck,
            decoder,
            target_layers=[2, 3, 4, 5, 6, 7, 8, 9],
            fuse_layer=[[0, 1, 2, 3, 4, 5, 6, 7]],
            remove_class_token=False,
            encoder_require_grad_layer=[],
            queries=None,
            queries_attn=None,
            noise=None,
    ) -> None:
        super(ASRB, self).__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder
        self.target_layers = target_layers
        self.fuse_layer = fuse_layer
        self.remove_class_token = remove_class_token
        self.encoder_require_grad_layer = encoder_require_grad_layer
        self.queries = queries[0]
        self.queries_attn = queries_attn
        self.noise = noise

        if not hasattr(self.encoder, 'num_register_tokens'):
            if hasattr(self.encoder, 'n_storage_tokens'):
                self.num_register_tokens = self.encoder.n_storage_tokens
            else:
                self.num_register_tokens = 0
        else:
            self.num_register_tokens = self.encoder.num_register_tokens


    def forward(self, x, decoder=True):
        x = self.encoder.prepare_tokens(x)
        B, L, _ = x.shape
        en_list = []
        for i, blk in enumerate(self.encoder.blocks):
            if i <= self.target_layers[-1]:
                if i in self.encoder_require_grad_layer:
                    x = blk(x)
                else:
                    with torch.no_grad():
                        x = blk(x)
            else:
                continue
            if i in self.target_layers:
                en_list.append(x)
        side = int(math.sqrt(en_list[0].shape[1] - 1 - self.num_register_tokens))

        if self.remove_class_token:
            en_list = [e[:, 1 + self.num_register_tokens:, :] for e in en_list]

        en = [self.fuse_feature([en_list[idx] for idx in idxs]) for idxs in self.fuse_layer]

        x = self.fuse_feature(en_list)

        queries = self.queries
        for blk in self.bottleneck:
            queries = blk(queries.unsqueeze(0).repeat((B, 1, 1)), x)

        if decoder:
            if isinstance(self.noise, nn.Module):
                x = self.noise(x)

            if isinstance(self.queries_attn, nn.ModuleList):
                for blk in self.queries_attn:
                    queries = blk(queries)

            de_list = []
            for i, block in enumerate(self.decoder):
                x = block(x, queries)
                de_list.append(x)
            de_list = de_list[::-1]

            de = [self.fuse_feature([de_list[idx] for idx in idxs]) for idxs in self.fuse_layer]

            if not self.remove_class_token:  # class tokens have not been removed above
                en = [e[:, 1 + self.num_register_tokens:, :] for e in en]
                de = [d[:, 1 + self.num_register_tokens:, :] for d in de]

            en = [e.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for e in en]
            de = [d.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for d in de]
            return en, de
        else:
            if not self.remove_class_token:  # class tokens have not been removed above
                x = x[:, 1 + self.num_register_tokens:, :]
            x = x.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous()
            return x, queries

    def fuse_feature(self, feat_list):
        return torch.stack(feat_list, dim=1).mean(dim=1)


