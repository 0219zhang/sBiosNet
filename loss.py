
import torch


def get_class_weight(y):
    """
    Input: y, torch.LongTensor
    Output: class_weight, torch.FloatTensor
    """
    _, class_counts = torch.unique(y, return_counts=True)
    class_prop = class_counts / len(y)
    class_weight = 1.0 / class_prop
    class_weight = class_weight / class_weight.min()
    return class_weight


def cross_entropy(
    pred,
    label,
    class_weight=None,
    sample_weight=None,
    reduction='mean'
):
    """
    Input:
        pred: torch.FloatTensor, [Batch, C]
        label: torch.FloatTensor, [Batch, ]
        class_weight: torch.FloatTensor, [C, ]
        sample_weight: torch.FloatTensor, [B, ]
        reduction: str
    Output:
        torch.FloatTensor, Loss
    """
    pred_log_sm = torch.nn.functional.log_softmax(pred, dim=1)
    samplewise_loss = -1 * torch.sum(label * pred_log_sm, dim=1)

    if sample_weight is not None:
        samplewise_loss *= sample_weight.squeeze()

    if class_weight is not None:
        class_weight = class_weight.to(label.device)
        class_weight = class_weight.repeat(samplewise_loss.size(0), 1)
        weight_vec, _ = torch.max(class_weight * label, dim=1)
        samplewise_loss = samplewise_loss * weight_vec

    if reduction == 'mean':
        loss = torch.mean(samplewise_loss)
    elif reduction == 'sum':
        loss = torch.sum(samplewise_loss)
    else:
        loss = samplewise_loss
    return loss
