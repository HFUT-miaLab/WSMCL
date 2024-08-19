import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from transformer import TransLayer


class TailFusion(nn.Module):
    def __init__(self, dim, n_classes):
        super(TailFusion, self).__init__()
        self.fc = nn.Linear(in_features=dim*2, out_features=dim)
        self.classifier = nn.Sequential(nn.Linear(in_features=dim*3, out_features=dim), nn.GELU(), nn.Linear(in_features=dim, out_features=n_classes))

    def forward(self, HE_slide_token, IHC_slide_token):

        add_fusion_token = torch.add(HE_slide_token, IHC_slide_token)
        multi_fusion_token = torch.mul(HE_slide_token, IHC_slide_token)
        concat_fusion_token = self.fc(torch.cat([HE_slide_token, IHC_slide_token], dim=-1))
        logits = self.classifier(torch.cat([add_fusion_token, multi_fusion_token, concat_fusion_token], dim=-1))
        return logits



class WSMCL(nn.Module):
    def __init__(self, input_dim=512, n_classes=4, k=10, return_atte=False):
        super(WSMCL, self).__init__()

        self.HE_FC = nn.Sequential(nn.Linear(in_features=input_dim, out_features=input_dim), nn.GELU())
        self.IHC_FC = nn.Sequential(nn.Linear(in_features=input_dim, out_features=input_dim), nn.GELU())

        self.HE_transformer = TransLayer(dim=input_dim, return_attn=True)
        self.IHC_transformer = TransLayer(dim=input_dim, return_attn=True)

        self.HE_classifier = nn.Sequential(nn.Linear(in_features=input_dim, out_features=n_classes))
        self.IHC_classifier = nn.Sequential(nn.Linear(in_features=input_dim, out_features=n_classes))
        self.MM_classifier = TailFusion(dim=512, n_classes=n_classes)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.return_atte = return_atte
        self.norm = nn.LayerNorm(512)
        self.k = k

    def forward(self, HE_features=None, IHC_features=None):

        result_dict = {}
        if HE_features is not None:
            HE_features = self.HE_FC(HE_features)
            HE_features = HE_features.unsqueeze(0)
            HE_A, HE_features = self.HE_transformer(HE_features)
            HE_features = self.norm(HE_features)
            HE_features = HE_features.squeeze(0)

            HE_select_top_index = torch.sort(HE_A, descending=True).indices[0][:self.k]
            HE_select_bottom_index = torch.sort(HE_A, descending=True).indices[0][-self.k:]
            HE_atte_top_features = torch.index_select(HE_features, 0, HE_select_top_index)
            HE_atte_bottom_features = torch.index_select(HE_features, 0, HE_select_bottom_index)
            HE_atte_features = torch.cat((HE_atte_top_features, HE_atte_bottom_features), dim=0)

            HE_slide_token = torch.mean(HE_features, dim=0, keepdim=True)
            HE_logit = self.HE_classifier(HE_slide_token)
            result_dict['HE_logit'] = HE_logit
            result_dict['HE_A'] = HE_A

        if IHC_features is not None:
            IHC_features = self.IHC_FC(IHC_features)
            IHC_features = IHC_features.unsqueeze(0)
            IHC_A, IHC_features = self.IHC_transformer(IHC_features)
            IHC_features = self.norm(IHC_features)
            IHC_features = IHC_features.squeeze(0)
            IHC_select_top_index = torch.sort(IHC_A, descending=True).indices[0][:self.k]
            IHC_select_bottom_index = torch.sort(IHC_A, descending=True).indices[0][-self.k:]
            IHC_atte_top_features = torch.index_select(IHC_features, 0, IHC_select_top_index)
            IHC_atte_bottom_features = torch.index_select(IHC_features, 0, IHC_select_bottom_index)
            IHC_atte_features = torch.cat((IHC_atte_top_features, IHC_atte_bottom_features), dim=0)
            IHC_slide_token = torch.mean(IHC_features, dim=0, keepdim=True)
            IHC_logit = self.IHC_classifier(IHC_slide_token)
            result_dict['IHC_logit'] = IHC_logit
            result_dict['IHC_A'] = IHC_A

        if HE_features is not None and IHC_features is not None:
            MM_logit = self.MM_classifier(HE_slide_token, IHC_slide_token)
            result_dict['MM_logit'] = MM_logit

            if self.training:
                result_dict['c_loss'] = self.contrast_loss(HE_atte_features, IHC_atte_features)

        return result_dict

    def contrast_loss(self, HE_atte_features, IHC_atte_features):

        logit_scale = self.logit_scale.exp()
        HE_atte_features = HE_atte_features / HE_atte_features.norm(dim=1, keepdim=True)
        IHC_atte_features = IHC_atte_features / IHC_atte_features.norm(dim=1, keepdim=True)
        logits_per_HE = logit_scale * HE_atte_features @ IHC_atte_features.t()
        labels = torch.tensor([0] * self.k).long().to(HE_atte_features.device)
        sum_loss = None

        for i in range(self.k):
            part1 = torch.cat([logits_per_HE[:self.k, i:i + 1], logits_per_HE[:self.k, -self.k:]], dim=1)
            part2 = torch.cat([logits_per_HE[-self.k:, self.k + i:self.k + i + 1], logits_per_HE[-self.k:, :self.k]],
                              dim=1)
            part3 = torch.cat([logits_per_HE.t()[:self.k, i:i + 1], logits_per_HE.t()[:self.k, -self.k:]], dim=1)
            part4 = torch.cat(
                [logits_per_HE.t()[-self.k:, self.k + i:self.k + i + 1], logits_per_HE.t()[-self.k:, :self.k]], dim=1)
            if sum_loss:
                sum_loss += (F.cross_entropy(part1, labels) +
                             F.cross_entropy(part2, labels) +
                             F.cross_entropy(part3, labels) +
                             F.cross_entropy(part4, labels)) / 4
            else:
                sum_loss = (F.cross_entropy(part1, labels) +
                            F.cross_entropy(part2, labels) +
                            F.cross_entropy(part3, labels) +
                            F.cross_entropy(part4, labels)) / 4
        sum_loss = sum_loss / self.k

        return sum_loss




