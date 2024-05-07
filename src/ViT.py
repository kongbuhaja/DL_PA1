from termcolor import colored
from typing import Dict
import copy

# PyTorch & Pytorch Lightning
from torch import nn
from torch.nn import Conv2d, Linear, Dropout, Softmax, LayerNorm
from torch.nn.functional import gelu
import torch
import math

class Embeddings(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.IMAGE_SIZE/cfg.PATCH_SIZE == cfg.IMAGE_SIZE/cfg.PATCH_SIZE

        n_patches = (cfg.IMAGE_SIZE//cfg.PATCH_SIZE)**2
        self.patch_embeddings = Conv2d(in_channels=3,
                                       out_channels=cfg.HIDDEN_SIZE,
                                       kernel_size=cfg.PATCH_SIZE,
                                       stride=cfg.PATCH_SIZE)
        self.cls_token = nn.Parameter(torch.zeros((1, 1, cfg.HIDDEN_SIZE)))
        self.position_embeddings = nn.Parameter(torch.zeros((1, n_patches+1, cfg.HIDDEN_SIZE)))

        self.dropout = Dropout(cfg.TRANSFORMER_DROPOUT_RATE)
        
    def forward(self, x):
        x = self.patch_embeddings(x)
        x = x.flatten(-2).transpose(-1, -2)
        x = torch.concat([x, self.cls_token.tile([x.shape[0], 1, 1])], -2)
        x = x + self.position_embeddings
        x = self.dropout(x)
        return x

class MSA(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.head_size = cfg.HEAD_SIZE
        self.attention_hidden_size = cfg.HIDDEN_SIZE // self.head_size
        self.hidden_size = self.attention_hidden_size * self.head_size

        self.query = Linear(cfg.HIDDEN_SIZE, self.hidden_size)
        self.key = Linear(cfg.HIDDEN_SIZE, self.hidden_size)
        self.value = Linear(cfg.HIDDEN_SIZE, self.hidden_size)

        self.out = Linear(cfg.HIDDEN_SIZE, cfg.HIDDEN_SIZE)
        self.attention_dropout = Dropout(cfg.MSA_DROPOUT_RATE)
        self.proj_dropout = Dropout(cfg.MSA_DROPOUT_RATE)

        self.softmax = Softmax(dim=-1)
    
    def attention_matrix(self, x):
        return x.view([*x.shape[:2], self.head_size, self.attention_hidden_size]).permute(0, 2, 1, 3)

    def forward(self, x):
        query = self.attention_matrix(self.query(x))
        key = self.attention_matrix(self.key(x))
        value = self.attention_matrix(self.value(x))

        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_size)
        probs = self.attention_dropout(self.softmax(scores))

        features = torch.matmul(probs, value).permute(0, 2, 1, 3).contiguous().view([*x.shape[:2], -1])
        out = self.proj_dropout(self.out(features))

        return out

class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.fc1 = Linear(cfg.HIDDEN_SIZE, cfg.MLP_SIZE)
        self.fc2 = Linear(cfg.MLP_SIZE, cfg.HIDDEN_SIZE)
        self.activate = gelu
        self.dropout = Dropout(cfg.TRANSFORMER_DROPOUT_RATE)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activate(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
    
class Transformer_block(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.msa_norm = LayerNorm(cfg.HIDDEN_SIZE)
        self.msa = MSA(cfg)

        self.ffn_norm = LayerNorm(cfg.HIDDEN_SIZE)
        self.ffn = MLP(cfg)
        
    def forward(self, x):
        residual = x
        x = self.msa_norm(x)
        x = self.msa(x)
        x = x + residual

        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + residual
        return x
    
class Transformer_encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        layers = []
        self.encoder_norm = LayerNorm(cfg.HIDDEN_SIZE)
        for _ in range(cfg.TRANSFORMER_SIZE):
            layers += [Transformer_block(cfg)]
        
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.encoder_norm(x)
        return x
    
class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.embeddings = Embeddings(cfg)
        self.encoder = Transformer_encoder(cfg)

    def forward(self, x):
        x = self.embeddings(x)
        x = self.encoder(x)
        return x
    
class VisionTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.transformer = Transformer(cfg)
        self.head = Linear(cfg.HIDDEN_SIZE, cfg.NUM_CLASSES)

    def forward(self, x):
        x = self.transformer(x)
        x = self.head(x[:, 0])
        return x