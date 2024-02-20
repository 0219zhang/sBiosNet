
import math
import copy

import torch
from torch import nn, autograd
from torch.nn import init
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np



class SparseLinear(nn.Module):
    """
    稀疏连接网络, 根据稀疏矩阵adj对FNN进行稀疏化
    adj: input * output
    """
    def __init__(self, adj, bias):
        super(SparseLinear, self).__init__()
        self.weight = Parameter(torch.empty(adj.T.shape))
        if bias:
            self.bias = Parameter(torch.empty(self.weight.size(0)))
        else:
            self.register_parameter('bias', None)
        self.adj = adj
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
        if not isinstance(self.adj, (torch.Tensor)):
            self.adj = torch.FloatTensor(self.adj)

    def forward(self, input):
        if self.adj.device != self.weight.device:
            self.adj = self.adj.to(self.weight.device)
        return F.linear(input, self.weight * self.adj.T, self.bias)


class SparseCell(nn.Module):
    """
    多层稀疏神经网络, 每层联合有Dropout, BatchNorm1d, ReLU
    """
    def __init__(self, maps, drop_p):
        super(SparseCell, self).__init__()
        # embeding
        embedding_layers = []
        for map in maps:
            embedding_layers += [
                nn.Dropout(p=drop_p),
                SparseLinear(adj=map, bias=True),
                nn.BatchNorm1d(map.size(1)),
                nn.ReLU(True)
            ]
        self.embedding_layers = nn.Sequential(*embedding_layers)

        self.embedding_num = maps[-1].size(1)
        self.classifier = nn.Linear(self.embedding_num, 2)

    def forward(self, x):
        embedding = self.embedding_layers(x)
        return self.classifier(embedding)


class DenseCell(nn.Module):
    """
    多层稀疏神经网络, 每层联合有Dropout, BatchNorm1d, ReLU
    """
    def __init__(self, maps, drop_p):
        super(DenseCell, self).__init__()
        # embeding
        embedding_layers = []
        for map in maps:
            embedding_layers += [
                nn.Dropout(p=drop_p),
                nn.Linear(map.size(0), map.size(1)),
                nn.BatchNorm1d(map.size(1)),
                nn.ReLU(True)
            ]
        self.embedding_layers = nn.Sequential(*embedding_layers)

        self.embedding_num = maps[-1].size(1)
        self.classifier = nn.Linear(self.embedding_num, 2)

    def forward(self, x):
        embedding = self.embedding_layers(x)
        return self.classifier(embedding)


class GradReverse(autograd.Function):
    """
    梯度上升
    weight: float, 在执行梯度上升时, 为梯度添加一个权重
    """
    @staticmethod
    def forward(ctx, x, weight):
        ctx.weight = weight
        return x.view_as(x) * 1.0

    @staticmethod
    def backward(ctx, grad_output):
        return (grad_output * -1. * ctx.weight), None


class DANN(nn.Module):
    """
    Domains Adversarial Neural Network
    """

    def __init__(
        self,
        model,
        hidden_node,
        n_domains
    ):
        super(DANN, self).__init__()
        self.model = model
        self.n_domains = n_domains
        self.embedding_layers = model.embedding_layers

        disc_in_nodes = [model.embedding_num] + hidden_node
        disc_out_nodes = hidden_node + [n_domains]
        discriminator_layers = []
        for n, (i, o) in enumerate(zip(disc_in_nodes, disc_out_nodes)):
            if n != len(disc_in_nodes) - 1:
                discriminator_layers += [
                    nn.Linear(i, o),
                    nn.ReLU(True)
                ]
            else:
                discriminator_layers += [nn.Linear(i, o)]
        self.discriminator_layers = nn.Sequential(*discriminator_layers)

    def set_rev_grad_weight(self, weight):
        self.weight = weight

    def forward(self, x):
        embedding = self.embedding_layers(x)
        embedding_rev = GradReverse.apply(embedding, self.weight)
        return self.discriminator_layers(embedding_rev)


class SampleMixUp(object):
    """
    样本对上的MixUp过程
    Input: sample, dict, keys=['cell_x', 'drug_x', 'output']
    """
    def __init__(self, alpha=0.2, keep_dominant_obs=True):
        self.alpha = alpha
        if alpha > 0.0:
            self.beta = torch.distributions.beta.Beta(self.alpha, self.alpha)
        self.keep_dominant_obs = keep_dominant_obs

    def __call__(self, cell_x, output):
        # 如果alpha=0.直接返回
        if self.alpha == .0:
            return cell_x, output
        else:
            # 对输入样本特征进行置换
            ridx = torch.randperm(cell_x.size(0))
            r_cell_x_ = cell_x[ridx]
            r_output = output[ridx]

            # 从beta分布中生成mixup的权重gamma
            gamma = self.beta.sample((cell_x.size(0), ))
            if self.keep_dominant_obs:
                gamma, _ = torch.max(
                    torch.stack([gamma, 1 - gamma], dim=1), dim=1
                )
            gamma = gamma.reshape(-1, 1)
            gamma = gamma.to(cell_x.device)

            # 通过mixup过程生成混合之后的样本特征
            mix_cell_x = self.mixup(cell_x, r_cell_x_, gamma=gamma)
            mix_output = self.mixup(output, r_output, gamma=gamma)

            return mix_cell_x, mix_output, ridx

    def mixup(self, x1, x2, gamma):
        return gamma * x1 + (1 - gamma) * x2


class InputDropout(object):
    """
    数据增强过程, 对输入样本数据随机Dropout 10%的样本
    """
    def __init__(self, p_drop=0.1):
        self.p_drop = p_drop

    def __call__(self, cell_x):
        # 生成原有样本的复制体
        au_cell_x = copy.deepcopy(cell_x)
        au_cell_x = torch.nn.functional.dropout(
            au_cell_x, p=self.p_drop, inplace=False
        )
        return au_cell_x


def sharpen_labels(q, T=0.5):
    """
    q: torch.FloatTensor
    T: float
    """
    if T == .0:
        _, idx = torch.max(q, dim=1)
        oh = torch.nn.functional.one_hot(idx, num_classes=q.size(1))
        return oh
    elif T == 1.0:
        return q
    else:
        q = torch.pow(q, 1.0 / T)
        q /= torch.sum(q, dim=1).reshape(-1, 1)
        return q


class PseudolabelModel(nn.Module):

    def __init__(
        self,
        sharpen_T=0.5,
        pseudolabel_min_confidence=0.0
    ):
        super(PseudolabelModel, self).__init__()
        self.sharpen_T = sharpen_T
        self.pseudolabel_min_confidence = pseudolabel_min_confidence
        self.teacher = None

    def __call__(self, model, unlabel_cell_x):
        # 当使用teacher模型生成样本伪标签时, 首先根据model的参数更新teacher模型
        self.teacher = copy.deepcopy(model)
        self.teacher = self.teacher.eval()
        pseudolabels = F.softmax(self.teacher(unlabel_cell_x), dim=1)

        # 提取伪标签的最大概率, 进而判断该伪标签是否为"高置信度的"
        highest_conf, _ = torch.max(pseudolabels, dim=1)
        confint = highest_conf >= self.pseudolabel_min_confidence

        # 对伪标签进行sharpen操作
        if self.sharpen_T is not None:
            pseudolabels = sharpen_labels(q=pseudolabels, T=self.sharpen_T)

        # 取消伪标签的梯度
        pseudolabels = pseudolabels.detach()

        # 返回对应样本的伪标签, 以及相应的置信度
        return pseudolabels, confint


# import pickle
# import numpy as np

# with open('datasets/predata.pkl', 'rb') as f:
#     predata = pickle.load(f)

# pdxs_data = predata['pdxs_data']
# pdxs_resp = predata['pdxs_resp']
# maps = predata['maps']

# pdxs_data = pdxs_data.loc[pdxs_resp.Model.values, :].values
# drug_data = np.random.randn(pdxs_data.shape[0], 1024)

# pdxs_data = torch.FloatTensor(pdxs_data)
# drug_data = torch.FloatTensor(drug_data)
# maps = [torch.FloatTensor(x.values) for x in maps]


# resp_model = ResponseModel(
#     maps, cell_dp=0.2, drug_hidden=[32, 32], drug_dp=0.2, clf_hidden=[32]
# )
# samples = {'cell_x': pdxs_data, 'drug_x': drug_data}

# mean_teacher = MeanTeacher()
# pesudolabels, confint = mean_teacher(resp_model, samples, epoch=1)


class ICLWeight(object):

    def __init__(
        self,
        ramp_epochs,
        burn_in_epochs=0,
        max_unsup_weight=10.0
    ):
        self.ramp_epochs = ramp_epochs
        self.burn_in_epochs = burn_in_epochs
        self.max_unsup_weight = max_unsup_weight

        if self.ramp_epochs == .0:
            self.step_size = self.max_unsup_weight
        else:
            self.step_size = self.max_unsup_weight / self.ramp_epochs

    def _get_weight(self, epoch):
        if epoch >= self.ramp_epochs + self.burn_in_epochs:
            weight = self.max_unsup_weight
        else:
            x = (epoch - self.burn_in_epochs) / self.ramp_epochs
            coef = np.exp(-5 * (x - 1) ** 2)
            weight = coef * self.max_unsup_weight
        return weight

    def __call__(self, epoch):
        if epoch < self.burn_in_epochs:
            weight = .0
        else:
            weight = self._get_weight(epoch)
        return weight


class SparseDAE(nn.Module):
    """
    多层稀疏神经网络, 每层联合有Dropout, BatchNorm1d, ReLU
    """
    def __init__(self, maps, drop_p):
        super(SparseDAE, self).__init__()
        # embeding layers
        embedding_layers = []
        for n, map in enumerate(maps):
            if n == 0:
                embedding_layers += [
                    nn.Dropout(p=drop_p),
                    SparseLinear(adj=map, bias=True),
                    nn.BatchNorm1d(map.size(1)),
                    nn.ReLU(True)
                ]
            else:
                embedding_layers += [
                    nn.Dropout(p=.0),
                    SparseLinear(adj=map, bias=True),
                    nn.BatchNorm1d(map.size(1)),
                    nn.ReLU(True)
                ]
        self.embedding_layers = nn.Sequential(*embedding_layers)

        # decoder layers
        decoder_layers = []
        for n, map in enumerate(maps[::-1]):
            if n == len(maps) - 1:
                decoder_layers += [nn.Linear(map.size(1), map.size(0))]
            else:
                decoder_layers += [
                    nn.Linear(map.size(1), map.size(0)),
                    nn.BatchNorm1d(map.size(0)),
                    nn.ReLU(True)
                ]
        self.decoder_layers = nn.Sequential(*decoder_layers)

    def forward(self, x):
        embedding = self.embedding_layers(x)
        return self.decoder_layers(embedding)
