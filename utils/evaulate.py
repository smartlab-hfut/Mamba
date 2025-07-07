import torch, math
from sklearn.metrics import roc_auc_score

def ece_score(confidence, correct, n_bins: int = 15):
    """简易 ECE 实现（equal-width bin）"""
    bins = torch.linspace(0, 1, n_bins + 1, device=confidence.device)
    ece = 0.
    for i in range(n_bins):
        mask = (confidence >= bins[i]) & (confidence < bins[i + 1])
        if mask.any():
            acc_bin  = correct[mask].float().mean()
            conf_bin = confidence[mask].mean()
            ece += mask.float().mean() * (acc_bin - conf_bin).abs()
    return ece.item()

@torch.no_grad()
def uq_metrics(logits_mc: torch.Tensor, labels: torch.Tensor):
    """
    logits_mc: [T, B, C]
    labels    : [B]
    """
    T, B, C = logits_mc.shape

    # --- 输出均值 ---
    logits_mean = logits_mc.mean(0)          # [B,C]
    prob_mean   = torch.softmax(logits_mean, -1)  # μ̂

    # --- 基本 Accuracy ---
    preds = prob_mean.argmax(1)
    acc   = (preds == labels).float().mean().item()

    # --- NLL ---
    nll = -prob_mean[torch.arange(B), labels].log().mean().item()

    # --- Brier ---
    one_hot = torch.nn.functional.one_hot(labels, C)
    brier = (prob_mean - one_hot.float()).pow(2).sum(1).mean().item()

    # --- ECE ---
    conf, _ = prob_mean.max(1)               # 置信度
    correct = preds == labels
    ece  = ece_score(conf, correct, n_bins=15)

    # --- AUROC (用最大 softmax 的负值当不确定度分数) ---
    # 1 = 错误, 0 = 正确
    err_mask = (~correct).cpu().numpy()
    auroc = roc_auc_score(err_mask, (-conf).cpu().numpy())

    return {
        "acc"   : acc,
        "nll"   : nll,
        "brier" : brier,
        "ece"   : ece,
        "auroc" : auroc,
    }



