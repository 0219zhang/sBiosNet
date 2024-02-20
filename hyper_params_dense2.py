
import time
import copy
import pickle
import argparse

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from model import DenseCell
from model import PseudolabelModel, SampleMixUp, ICLWeight
from loss import cross_entropy


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
    labeled_loader,
    unlabeled_loader,
    mixup_alpha,
    sharpen_T,
    pseudolabel_min_confidence,
    ul_icl_ramp_epochs,
    ul_icl_burn_in_epoch,
    ul_icl_max_unsup_weight,
    learn_rate,
    epochs,
    min_epochs,
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

    mixup_op = SampleMixUp(alpha=mixup_alpha)
    teacher_model = PseudolabelModel(
        sharpen_T=sharpen_T,
        pseudolabel_min_confidence=pseudolabel_min_confidence
    )
    iter_unlabeled_dl = iter(unlabeled_loader)

    unsup_icl_weight = ICLWeight(
        ramp_epochs=ul_icl_ramp_epochs,
        burn_in_epochs=ul_icl_burn_in_epoch,
        max_unsup_weight=ul_icl_max_unsup_weight
    )
    unsup_criterion = nn.MSELoss(reduction='none')
    sup_criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        resp_model.parameters(), lr=learn_rate, weight_decay=weight_decay
    )
    optimizer_step = StepLR(
        optimizer=optimizer, step_size=lr_step_size, gamma=lr_gamma
    )

    best_auc = .0
    no_improve = 0
    best_wt = copy.deepcopy(resp_model.state_dict())

    for epoch in tqdm(range(epochs), leave=False):
        # Train phase
        resp_model.train()
        ytrue, ypred = [], []

        for l_cellx, l_resp in labeled_loader['train']:
            l_cellx = l_cellx.to(device)
            l_resp = l_resp.to(device)
            l_resp_oh = F.one_hot(l_resp, num_classes=2).squeeze().float()

            try:
                ul_cellx, ul_resp = next(iter_unlabeled_dl)
            except StopIteration:
                iter_unlabeled_dl = iter(unlabeled_loader)
                ul_cellx, ul_resp = next(iter_unlabeled_dl)

            ul_cellx = ul_cellx.to(device)
            ul_resp = ul_resp.to(device)

            pseudolabel, confint = teacher_model(
                model=resp_model, unlabel_cell_x=ul_cellx
            )

            conf_ul_cellx = ul_cellx[confint]
            conf_pseudolabel = pseudolabel[confint]

            ucnf_ul_cellx = ul_cellx[~confint]
            ucnf_pseudolabel = pseudolabel[~confint]

            cat_cellx = torch.cat([conf_ul_cellx, l_cellx], dim=0)
            cat_resp = torch.cat([conf_pseudolabel, l_resp_oh], dim=0)

            mix_cellx, mix_resp, _ = mixup_op(cat_cellx, cat_resp)

            n_confinted = conf_ul_cellx.size(0)

            mix_ul_cellx_ = mix_cellx[:n_confinted]
            mix_ul_resp = mix_resp[:n_confinted]

            final_ul_cellx_ = torch.cat(
                [mix_ul_cellx_, ucnf_ul_cellx], dim=0
            )
            final_ul_resp = torch.cat(
                [mix_ul_resp, ucnf_pseudolabel], dim=0
            )

            final_ul_output = F.softmax(resp_model(final_ul_cellx_), dim=1)

            unsup_loss = unsup_criterion(final_ul_output, final_ul_resp)
            if unsup_loss.dim() > 1:
                unsup_loss = torch.sum(unsup_loss, dim=1)

            scale_vec = torch.zeros_like(unsup_loss).float()
            scale_vec = scale_vec.to(unsup_loss.device)
            scale_vec[:n_confinted] += 1.
            unsup_loss *= scale_vec
            unsup_loss = torch.mean(unsup_loss) * unsup_icl_weight(epoch)

            mix_l_cellx_ = mix_cellx[n_confinted:]
            mix_l_resp = mix_resp[n_confinted:]
            mix_l_output = resp_model(mix_l_cellx_)
            sup_loss = cross_entropy(
                mix_l_output, mix_l_resp, reduction='mean'
            )

            total_loss = sup_loss + unsup_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        # Valid Phase
        resp_model.eval()
        with torch.no_grad():
            ytrue, ypred = [], []

            for l_cellx, l_resp in labeled_loader['valid']:
                l_cellx = l_cellx.to(device)
                l_resp = l_resp.to(device)
                l_output = resp_model(l_cellx)

                sup_loss = sup_criterion(l_output, l_resp)

                ytrue.append(l_resp)
                ypred.append(l_output)

            with torch.no_grad():
                ytrue = torch.cat(ytrue, dim=0).cpu().numpy()
                ypred = F.softmax(torch.cat(ypred, dim=0), dim=1).cpu().numpy()
                auroc = roc_auc_score(ytrue, ypred[:, 1])

                if epoch >= min_epochs:
                    if auroc > best_auc:
                        best_auc = auroc
                        best_wt = copy.deepcopy(resp_model.state_dict())
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve = 0
        optimizer_step.step()
        if no_improve == early_stop:
            break

    # Predict Phase
    with torch.no_grad():
        resp_model.load_state_dict(best_wt)
        resp_model.eval()
        best_wt = copy.deepcopy(resp_model).cpu().state_dict()

        hyper_ytrue, hyper_ypred = [], []
        for ul_cellx, ul_resp in labeled_loader['hyper']:
            ul_cellx = ul_cellx.to(device)
            output = resp_model(ul_cellx)
            hyper_ytrue.append(ul_resp)
            hyper_ypred.append(output)

        hyper_ytrue = torch.cat(hyper_ytrue, dim=0)
        hyper_ypred = torch.softmax(
            torch.cat(hyper_ypred, dim=0), dim=1
        ).cpu().numpy()
        hyper_auroc = roc_auc_score(hyper_ytrue, hyper_ypred[:, 1])

        test_ytrue, test_ypred = [], []
        for ul_cellx, ul_resp in labeled_loader['test']:
            ul_cellx = ul_cellx.to(device)
            output = resp_model(ul_cellx)
            test_ytrue.append(ul_resp)
            test_ypred.append(output)

        test_ytrue = torch.cat(test_ytrue, dim=0)
        test_ypred = torch.softmax(
            torch.cat(test_ypred, dim=0), dim=1
        ).cpu().numpy()
        test_pred_result = pd.DataFrame({
            'ytrue': test_ytrue, 'tpred': test_ypred[:, 1]
        })
        test_auroc = roc_auc_score(test_ytrue, test_ypred[:, 1])
        return hyper_auroc, test_auroc, test_pred_result, best_wt


def main():
    # 模型超参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_id', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seeds', type=int, default=1776)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--min_epochs', type=int, default=0)
    parser.add_argument('--early_stop', type=int, default=40)
    parser.add_argument('--path', type=str, default='test0')
    parser.add_argument('--learn_rate', type=float, default=0.01)
    #
    parser.add_argument('--mixup_alpha', default=[0.35, 0.15])
    parser.add_argument('--sharpen_T', default=[0.95])
    parser.add_argument('--pseudolabel_min_conf', default=[0.6, 0.7])
    parser.add_argument('--ul_icl_max_unsup_weight', default=[2.5, 3.5, 4.5])
    parser.add_argument('--ul_icl_ramp_epochs', default=[20, 30])
    parser.add_argument('--ul_icl_burn_in_epochs', default=[10])
    parser.add_argument('--lr_step_size', default=[30])
    parser.add_argument('--lr_gamma', default=[0.9, 0.75])
    parser.add_argument('--weight_decay', default=[1e-3, 1e-4])
    parser.add_argument('--cell_dp', default=[0.2, 0.1])
    args = parser.parse_args()

    device = torch.device('cuda:' + str(args.cuda_id))
    batch_size = args.batch_size
    seeds = args.seeds
    epochs = args.epochs
    min_epochs = args.min_epochs
    early_stop = args.early_stop
    learn_rate = args.learn_rate

    #
    param_grid = ParameterGrid({
        'mixup_alpha': args.mixup_alpha,
        'sharpen_T': args.sharpen_T,
        'pseudolabel_min_confidence': args.pseudolabel_min_conf,
        'ul_icl_ramp_epochs': args.ul_icl_ramp_epochs,
        'ul_icl_burn_in_epochs': args.ul_icl_burn_in_epochs,
        'ul_icl_max_unsup_weight': args.ul_icl_max_unsup_weight,
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

    maps = [
        torch.zeros((755, 19), dtype=torch.float32),
        torch.zeros((19, 10), dtype=torch.float32),
        torch.zeros((10, 6), dtype=torch.float32)
    ]

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
    unlabel_loader = DataLoader(
        CellDrugData(
            torch.FloatTensor(miao_mut), torch.LongTensor(miao_resp)
        ),
        batch_size=batch_size, shuffle=True,
        drop_last=miao_mut.shape[0] % batch_size == 1
    )

    result = {
        'mixup_alpha': [], 'sharpen_T': [], 'pseudolabel_min_confidence': [],
        'ramp_epochs': [], 'burn_in_epochs': [], 'max_unsup_weight': [],
        'weight_decay': [], 'lr_step_size': [], 'lr_gamma': [],
        'cell_dp': [], 'hyper_auc': [], 'test_auc': [], 'time': []
    }
    best_hyper_auc = .0
    for param in tqdm(param_grid, total=len(list(param_grid))):
        mixup_alpha = param['mixup_alpha']
        sharpen_T = param['sharpen_T']
        pseudolabel_min_confidence = param['pseudolabel_min_confidence']
        ul_icl_ramp_epochs = param['ul_icl_ramp_epochs']
        ul_icl_burn_in_epochs = param['ul_icl_burn_in_epochs']
        ul_icl_max_unsup_weight = param['ul_icl_max_unsup_weight']
        weight_decay = param['weight_decay']
        lr_step_size = param['lr_step_size']
        lr_gamma = param['lr_gamma']
        cell_dp = param['cell_dp']

        torch.manual_seed(seeds)
        resp_model = DenseCell(maps=maps, drop_p=cell_dp).to(device)
        with open('rawdata/dense2_pretrain_wt.pkl', 'rb') as f:
            wt = pickle.load(f)
            resp_model.load_state_dict(wt)

        start = time.perf_counter()
        hyper_auc, test_auc, pred_df, best_wt = train_eval_model(
            resp_model=resp_model,
            labeled_loader=total_loader,
            unlabeled_loader=unlabel_loader,
            mixup_alpha=mixup_alpha,
            sharpen_T=sharpen_T,
            pseudolabel_min_confidence=pseudolabel_min_confidence,
            ul_icl_ramp_epochs=ul_icl_ramp_epochs,
            ul_icl_burn_in_epoch=ul_icl_burn_in_epochs,
            ul_icl_max_unsup_weight=ul_icl_max_unsup_weight,
            learn_rate=learn_rate,
            epochs=epochs,
            min_epochs=min_epochs,
            early_stop=early_stop,
            weight_decay=weight_decay,
            lr_step_size=lr_step_size,
            lr_gamma=lr_gamma,
            device=device
        )
        end = time.perf_counter()
        result['hyper_auc'].append(hyper_auc)
        result['test_auc'].append(test_auc)
        result['burn_in_epochs'].append(ul_icl_burn_in_epochs)
        result['cell_dp'].append(cell_dp)
        result['lr_gamma'].append(lr_gamma)
        result['lr_step_size'].append(lr_step_size)
        result['max_unsup_weight'].append(ul_icl_max_unsup_weight)
        result['mixup_alpha'].append(mixup_alpha)
        result['pseudolabel_min_confidence'].append(pseudolabel_min_confidence)
        result['weight_decay'].append(weight_decay)
        result['ramp_epochs'].append(ul_icl_ramp_epochs)
        result['sharpen_T'].append(sharpen_T)
        result['time'].append(end - start)
        hyper_result = pd.DataFrame(result).sort_values(
            by=['hyper_auc', 'time'], ascending=[False, True]
        )
        best_params = {
            'batch_size': args.batch_size,
            'seeds': args.seeds,
            'epochs': args.epochs,
            'min_epochs': args.min_epochs,
            'early_stop': args.early_stop,
            'learn_rate': args.learn_rate,
            'burn_in_epochs': ul_icl_burn_in_epochs,
            'cell_dp': cell_dp,
            'lr_gamma': lr_gamma,
            'lr_step_size': lr_step_size,
            'max_unsup_weight': ul_icl_max_unsup_weight,
            'mixup_alpha': mixup_alpha,
            'pseudolabel_min_confidence': pseudolabel_min_confidence,
            'weight_decay': weight_decay,
            'ramp_epochs': ul_icl_ramp_epochs,
            'sharpen_T': sharpen_T
        }
        hyper_result.to_csv(
            'hyper_params_search/dense2_param_result.csv', index=None
        )
        if hyper_auc > best_hyper_auc:
            best_hyper_auc = hyper_auc
            best_time = end - start
            pred_df.to_csv('hyper_params_search/dense2_pred_df.csv', index=None)
            with open('hyper_params_search/dense2_best_wt.pkl', 'wb') as f:
                pickle.dump(best_wt, f)
            with open('hyper_params_search/dense2_best_params.pkl', 'wb') as f:
                pickle.dump(best_params, f)
        elif hyper_auc == best_hyper_auc:
            if best_time > end - start:
                best_time = end - start
                pred_df.to_csv(
                    'hyper_params_search/dense2_pred_df.csv', index=None
                )
                with open('hyper_params_search/dense2_best_wt.pkl', 'wb') as f:
                    pickle.dump(best_wt, f)
                with open(
                    'hyper_params_search/dense2_best_params.pkl', 'wb'
                ) as f:
                    pickle.dump(best_params, f)



if __name__ == '__main__':
    main()

