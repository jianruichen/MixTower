# ------------------------------------------------------------------------------ #
__author__ = 'Deguang Chen'
# Description: the definition of the improved MCAN
# ------------------------------------------------------------------------------ #

import torch
from torch import nn
from torch.nn import functional as F
import math
from transformers import AutoModel, logging
from transformers import BertModel
logging.set_verbosity_error()

from .layers import *
from .net_utils import *


class MixLayer(nn.Module):
    def __init__(self, __C, layer_num):
        super().__init__()
        self.__C = __C
        self.layer_num = layer_num
        self.sa_t = eval(__C.ARCH_CEIL[0])(__C)
        self.ffn_t = eval(__C.ARCH_CEIL[1])(__C)
        self.sa_v = eval(__C.ARCH_CEIL[2])(__C)
        self.ffn_v = eval(__C.ARCH_CEIL[3])(__C)
        self.sa_m = eval(__C.ARCH_CEIL[4])(__C)
        self.ffn_m = eval(__C.ARCH_CEIL[5])(__C)

    def forward(self, lang_feat, img_feat, mix_feat, lang_feat_mask, img_feat_mask, mix_feat_mask):
        lang_feat = self.sa_t(lang_feat, lang_feat_mask)
        lang_feat = self.ffn_t(lang_feat)
        img_feat = self.sa_v(img_feat, img_feat_mask)
        img_feat = self.ffn_v(img_feat)
        mix_feat = self.sa_m(mix_feat, mix_feat_mask)
        mix_feat = self.ffn_m(mix_feat)
        return lang_feat, img_feat, mix_feat


# The definition of the encoder-decoder backbone of MCAN.
class MixEncoder(nn.Module):
    def __init__(self, __C):
        super().__init__()
        self.mix_layer = nn.ModuleList(MixLayer(__C, i) for i in range(__C.LAYER))
        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = nn.LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, lang_feat, img_feat, mix_feat, lang_feat_mask, img_feat_mask, mix_feat_mask):
        for enc_layer in self.mix_layer:
            lang_feat, img_feat, mix_feat = enc_layer(lang_feat, img_feat, mix_feat, lang_feat_mask, img_feat_mask, mix_feat_mask)
            img_lang_mix_feat = torch.cat((img_feat, lang_feat), dim=-2)
            mix_feat += img_lang_mix_feat
            mix_feat = self.norm1(mix_feat + self.dropout1(mix_feat))
        return lang_feat, img_feat, mix_feat


class MCAN(nn.Module):
    def __init__(self, __C, answer_size):
        super().__init__()

        # answer_size = trainset.ans_size
        self.__C = __C
        self.bert = BertModel.from_pretrained(__C.LARGE_BERT)

        self.lang_adapt = nn.Sequential(nn.Linear(__C.LANG_FEAT_SIZE, __C.HIDDEN_SIZE), nn.Tanh(),)
        self.img_adapt = nn.Sequential(nn.Linear(__C.IMG_FEAT_SIZE, __C.HIDDEN_SIZE, bias=False),)

        self.backbone = MixEncoder(__C)
        self.attflat_lang = AttFlat(__C)
        self.attflat_img = AttFlat(__C)
        self.attflat_mix = AttFlat(__C)
        self.proj_norm = nn.LayerNorm(__C.FLAT_OUT_SIZE)
        self.proj = nn.Linear(__C.FLAT_OUT_SIZE, answer_size)

    def forward(self, input_tuple, output_answer_latent=False):
        img_feat, ques_ix = input_tuple

        # Make mask
        lang_feat_mask = self.make_mask(ques_ix.unsqueeze(2))
        img_feat_mask = self.make_mask(img_feat)

        # Pre-process Language Feature
        lang_feat = self.bert(ques_ix, attention_mask= ~lang_feat_mask.squeeze(1).squeeze(1))[0]
        lang_feat = self.lang_adapt(lang_feat)

        # Pre-process Image Feature
        img_feat = self.img_adapt(img_feat)

        # Backbone Framework
        mix_feat = torch.cat((img_feat, lang_feat), dim=-2)
        mix_feat_mask = torch.cat((img_feat_mask, lang_feat_mask), dim=-1)
        lang_feat, img_feat, mix_feat = self.backbone(lang_feat, img_feat, mix_feat, lang_feat_mask, img_feat_mask, mix_feat_mask)

        lang_feat = self.attflat_lang(lang_feat, lang_feat_mask)
        img_feat = self.attflat_img(img_feat, img_feat_mask)
        mix_feat = self.attflat_mix(mix_feat, mix_feat_mask)

        proj_feat = lang_feat + mix_feat
        answer_latent = self.proj_norm(proj_feat)
        proj_feat = self.proj(answer_latent)

        if output_answer_latent:
            return proj_feat, answer_latent

        return proj_feat

    # Masking
    def make_mask(self, feature):
        return (torch.sum(torch.abs(feature), dim=-1) == 0).unsqueeze(1).unsqueeze(2)
