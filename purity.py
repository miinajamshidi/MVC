import torch
def purity_score(y_true, y_pred):
    unique_labels = torch.unique(y_pred)
    y_pred_to_true = torch.zeros(len(unique_labels), dtype=y_true.dtype)

    for true_label, pred_label in zip(y_true, y_pred):
        pred_label_idx = (pred_label == unique_labels).nonzero().item()
        y_pred_to_true[pred_label_idx] = true_label

    correct = sum(1 for true_label, pred_label in zip(y_true, y_pred) if true_label == y_pred_to_true[(pred_label == unique_labels).nonzero().item()])

    return correct / len(y_pred)