
import pickle
import numpy as np
import pandas as pd
from captum.attr import DeepLift
from captum.attr import LayerConductance
import torch

from model import SparseCell



with open('rawdata/predata.pkl', 'rb') as f:
    predata = pickle.load(f)

data = predata['data']
clin = predata['clin']
maps = predata['maps']
maps[0].to_csv('deeplift/input2gene_adj.csv')
maps[1].to_csv('deeplift/gene2pathc_adj.csv')
maps[2].to_csv('deeplift/pathc2pathb_adj.csv')

input_gene_name = maps[0].index.values
gene_name = maps[0].columns.values
pc_name = maps[1].columns.values
pb_name = maps[2].columns.values


model_maps = [torch.FloatTensor(x.values) for x in maps]

with open('hyper_params_search/best_params.pkl', 'rb') as f:
    params = pickle.load(f)

device = torch.device('cuda:1')
seeds = params['seeds']
cell_dp = params['cell_dp']
resp_model = SparseCell(maps=model_maps, drop_p=cell_dp).to(device)

with open('hyper_params_search/best_wt.pkl', 'rb') as f:
    best_wt = pickle.load(f)

resp_model.load_state_dict(best_wt)
resp_model.eval()
x = data.loc[clin.dataset.values == 'd2', :].values

#
deeplift = DeepLift(resp_model)
x = torch.FloatTensor(x).to(device).requires_grad_()

# 输入节点的重要性
input_attr = deeplift.attribute(x, target=1, return_convergence_delta=False)
input_attr = input_attr.detach().cpu().numpy()
input_attr = pd.DataFrame(input_attr, columns=input_gene_name).T
input_attr.loc[:, 'vim'] = input_attr.mean(axis=1)
input_attr.to_csv('deeplift/input_attr.csv')

# 基因节点的重要性
cond_gene = LayerConductance(resp_model, resp_model.embedding_layers[1])
gene_attr = cond_gene.attribute(x, target=1).detach().cpu().numpy()
gene_attr = pd.DataFrame(gene_attr, columns=gene_name).T
gene_attr.loc[:, 'vim'] = gene_attr.mean(axis=1)
pd.DataFrame(gene_attr).to_csv('deeplift/gene_attr.csv')

# 通路C节点的重要性
cond_pathc = LayerConductance(resp_model, resp_model.embedding_layers[5])
pathc_attr = cond_pathc.attribute(x, target=1).detach().cpu().numpy()
pathc_attr = pd.DataFrame(pathc_attr, columns=pc_name).T
pathc_attr.loc[:, 'vim'] = pathc_attr.mean(axis=1)
pathc_attr.to_csv('deeplift/pathc_attr.csv', index=None)

# 通路B节点的重要性
cond_pathb = LayerConductance(resp_model, resp_model.embedding_layers[9])
pathb_attr = cond_pathb.attribute(x, target=1).detach().cpu().numpy()
pathb_attr = pd.DataFrame(pathb_attr, columns=pb_name).T
pathb_attr.loc[:, 'vim'] = pathb_attr.mean(axis=1)
pathb_attr.to_csv('deeplift/pathb_attr.csv', index=None)

# input2gene wt
wt0 = resp_model.embedding_layers[1].weight.T.detach().cpu().numpy()
input2gene = maps[0] * wt0
input2gene.to_csv('deeplift/input2gene_wt.csv')

# gene2pathc wt
wt1 = resp_model.embedding_layers[5].weight.T.detach().cpu().numpy()
gene2pathc = maps[1] * wt1
gene2pathc.to_csv('deeplift/gene2pathc_wt.csv')

# pathc2pathb wt
wt2 = resp_model.embedding_layers[9].weight.T.detach().cpu().numpy()
pathc2pathb = maps[2] * wt2
pathc2pathb.to_csv('deeplift/pathc2pathb_wt.csv')

pathb2output = resp_model.classifier.weight.T.detach().cpu().numpy()
pathb2output = pd.DataFrame(
    pathb2output, index=maps[-1].columns.values
)
pathb2output.to_csv('deeplift/pathb2output.csv')










import pickle


with open("D:\data\doctor\肺癌-免疫治疗-半监督\data_from_frp70\hyper_params_search\best_params.pkl", 'rb') as f:
    param = pickle.load(f)

