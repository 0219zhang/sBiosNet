
import time
import copy
import pickle
import argparse

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from model import SparseCell


class CellDrugData(Dataset):

    def __init__(self, omics_data, resp_label):
        super(CellDrugData, self).__init__()
        self.omics_data = omics_data
        self.resp_label = resp_label

    def __len__(self):
        return len(self.resp_label)

    def __getitem__(self, idx):
        omics_x = self.omics_data[idx, :]
        resp_x = self.resp_label[idx]
        return omics_x, resp_x


def train_eval_model(
    resp_model,
    dataloader,
    learn_rate,
    epochs,
    early_stop,
    weight_decay,
    lr_step_size,
    lr_gamma,
    device
):
    """
    resp_model: 药物敏感性预测模型, 输入多组学数据和药物结构特征
    labeled_loader: 有标签样本的数量
    """

    if early_stop is None:
        early_stop = epochs

    criterion = nn.CrossEntropyLoss(reduction='mean')
    optimizer = optim.Adam(
        resp_model.parameters(), lr=learn_rate, weight_decay=weight_decay
    )
    optimizer_step = StepLR(
        optimizer=optimizer, step_size=lr_step_size, gamma=lr_gamma
    )

    best_auc = .0
    no_improve = 0
    best_wt = copy.deepcopy(resp_model.state_dict())

    for _ in tqdm(range(epochs), leave=False):
        ytrue, ypred = [], []
        for phase in ['train', 'valid']:
            if phase == 'train':
                resp_model.train()
            else:
                resp_model.eval()

            with torch.set_grad_enabled(phase == 'train'):
                for cellx, resp in dataloader[phase]:
                    cellx, resp = cellx.to(device), resp.to(device)
                    output = resp_model(cellx)
                    loss = criterion(output, resp)

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    ytrue.append(resp)
                    ypred.append(output)

        with torch.no_grad():
            if phase == 'valid':
                ytrue = torch.cat(ytrue, dim=0).cpu().numpy()
                ypred = F.softmax(torch.cat(ypred, dim=0), dim=1).cpu().numpy()
                auroc = roc_auc_score(ytrue, ypred[:, 1])
                if auroc > best_auc:
                    best_auc = auroc
                    best_wt = copy.deepcopy(resp_model.state_dict())
                    no_improve = 0
                else:
                    no_improve += 1

        optimizer_step.step()
        if no_improve == early_stop:
            break

    # Predict Phase
    with torch.no_grad():
        resp_model.load_state_dict(best_wt)
        resp_model.eval()

        hyper_ytrue, hyper_ypred = [], []
        for cellx, resp in dataloader['hyper']:
            cellx = cellx.to(device)
            output = resp_model(cellx)
            hyper_ytrue.append(resp)
            hyper_ypred.append(output)

        hyper_ytrue = torch.cat(hyper_ytrue, dim=0)
        hyper_ypred = torch.softmax(
            torch.cat(hyper_ypred, dim=0), dim=1
        ).cpu().numpy()
        hyper_auroc = roc_auc_score(hyper_ytrue, hyper_ypred[:, 1])

        test_ytrue, test_ypred = [], []
        for cellx, resp in dataloader['test']:
            cellx = cellx.to(device)
            output = resp_model(cellx)
            test_ytrue.append(resp)
            test_ypred.append(output)

        test_ytrue = torch.cat(test_ytrue, dim=0)
        test_ypred = torch.softmax(
            torch.cat(test_ypred, dim=0), dim=1
        ).cpu().numpy()
        test_pred_result = pd.DataFrame({
            'ytrue': test_ytrue, 'tpred': test_ypred[:, 1]
        })
        test_auroc = roc_auc_score(test_ytrue, test_ypred[:, 1])
        # print('AUC: {:.4f}'.format(auroc))
        return hyper_auroc, test_auroc, test_pred_result, best_wt


def main():
    # 模型超参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_id', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seeds', type=int, default=1776)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--early_stop', type=int, default=40)
    parser.add_argument('--path', type=str, default='test0')
    parser.add_argument('--learn_rate', type=float, default=0.01)
    #
    parser.add_argument('--lr_step_size', default=[30])
    parser.add_argument('--lr_gamma', default=[0.9, 0.75])
    parser.add_argument('--weight_decay', default=[1e-3, 1e-4])
    parser.add_argument('--cell_dp', default=[0.2, 0.1])
    args = parser.parse_args()

    device = torch.device('cuda:' + str(args.cuda_id))
    writer = SummaryWriter(log_dir='path/semi_' + args.path)
    batch_size = args.batch_size
    seeds = args.seeds
    epochs = args.epochs
    early_stop = args.early_stop
    learn_rate = args.learn_rate

    #
    param_grid = ParameterGrid({
        'weight_decay': args.weight_decay,
        'lr_step_size': args.lr_step_size,
        'lr_gamma': args.lr_gamma,
        'cell_dp': args.cell_dp
    })

    with open('rawdata/predata.pkl', 'rb') as f:
        predata = pickle.load(f)

    data, clin = predata['data'], predata['clin']
    rizvi_mut = data.loc[clin.dataset.values == 'd1', :].values
    rizvi_resp = clin.resp.values[clin.dataset.values == 'd1']
    maps = [torch.FloatTensor(x.values) for x in predata['maps']]

    miao_mut = data.loc[clin.dataset.values == 'd2', :].values
    miao_resp = clin.resp.values[clin.dataset.values == 'd2']

    trainx, validx, trainy, validy = train_test_split(
        rizvi_mut, rizvi_resp, test_size=0.1, shuffle=True,
        random_state=seeds, stratify=rizvi_resp
    )
    trainx, hyperx, trainy, hypery = train_test_split(
        trainx, trainy, test_size=len(validy), shuffle=True,
        random_state=seeds, stratify=trainy
    )

    total_loader = {
        'train': DataLoader(
            CellDrugData(
                torch.FloatTensor(trainx), torch.LongTensor(trainy)
            ),
            batch_size=batch_size, shuffle=True,
            drop_last=trainx.shape[0] % batch_size == 1
        ),
        'valid': DataLoader(
            CellDrugData(
                torch.FloatTensor(validx), torch.LongTensor(validy)
            ),
            batch_size=batch_size
        ),
        'hyper': DataLoader(
            CellDrugData(
                torch.FloatTensor(hyperx), torch.LongTensor(hypery)
            ),
            batch_size=batch_size
        ),
        'test': DataLoader(
            CellDrugData(
                torch.FloatTensor(miao_mut), torch.LongTensor(miao_resp)
            ),
            batch_size=batch_size
        )
    }

    result = {
        'weight_decay': [], 'lr_step_size': [], 'lr_gamma': [],
        'cell_dp': [], 'hyper_auc': [], 'test_auc': [], 'time': []
    }
    best_hyper_auc = .0
    for param in tqdm(param_grid, total=len(list(param_grid))):
        weight_decay = param['weight_decay']
        lr_step_size = param['lr_step_size']
        lr_gamma = param['lr_gamma']
        cell_dp = param['cell_dp']

        torch.manual_seed(seeds)
        resp_model = SparseCell(maps=maps, drop_p=cell_dp).to(device)
        with open('rawdata/pretrain_wt.pkl', 'rb') as f:
            wt = pickle.load(f)
            resp_model.load_state_dict(wt)
        # resp_model = DenseCell(maps=maps, drop_p=cell_dp).to(device)

        start = time.perf_counter()
        hyper_auc, test_auc, pred_df, _ = train_eval_model(
            resp_model=resp_model,
            dataloader=total_loader,
            learn_rate=learn_rate,
            epochs=epochs,
            early_stop=early_stop,
            weight_decay=weight_decay,
            lr_step_size=lr_step_size,
            lr_gamma=lr_gamma,
            device=device
        )
        end = time.perf_counter()
        result['hyper_auc'].append(hyper_auc)
        result['test_auc'].append(test_auc)
        result['cell_dp'].append(cell_dp)
        result['lr_gamma'].append(lr_gamma)
        result['lr_step_size'].append(lr_step_size)
        result['weight_decay'].append(weight_decay)
        result['time'].append(end - start)
        hyper_result = pd.DataFrame(result).sort_values(
            by=['hyper_auc', 'time'], ascending=[False, True]
        )
        best_params = {
            'batch_size': args.batch_size,
            'seeds': args.seeds,
            'epochs': args.epochs,
            'early_stop': args.early_stop,
            'learn_rate': args.learn_rate,
            'cell_dp': cell_dp,
            'lr_gamma': lr_gamma,
            'lr_step_size': lr_step_size,
            'weight_decay': weight_decay
        }
        hyper_result.to_csv(
            'hyper_params_search/supv_param_result.csv', index=None
        )
        if hyper_auc > best_hyper_auc:
            best_hyper_auc = hyper_auc
            best_time = end - start
            pred_df.to_csv('hyper_params_search/supv_pred_df.csv', index=None)
            with open('hyper_params_search/supv_best_params.pkl', 'wb') as f:
                pickle.dump(best_params, f)
        elif hyper_auc == best_hyper_auc:
            if best_time > end - start:
                best_time = end - start
                pred_df.to_csv(
                    'hyper_params_search/supv_pred_df.csv', index=None
                )
                with open(
                    'hyper_params_search/supv_best_params.pkl', 'wb'
                ) as f:
                    pickle.dump(best_params, f)


if __name__ == '__main__':
    main()

