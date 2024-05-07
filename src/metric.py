from torchmetrics import Metric
import torch

class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target):
        max_id = torch.argmax(preds, axis=-1)
        assert max_id.shape == target.shape

        correct = torch.count_nonzero(max_id==target)
        self.correct += correct
        self.total += target.numel()

    def compute(self, eps=1e-6):
        return self.correct.float() / (self.total.float() + eps)
    
class MyF1Score(Metric):
    def __init__(self, num_classes):
        super().__init__()
        self.add_state('TP', default=torch.zeros((num_classes)), dist_reduce_fx='sum')
        self.add_state('FP', default=torch.zeros((num_classes)), dist_reduce_fx='sum')
        self.add_state('FN', default=torch.zeros((num_classes)), dist_reduce_fx='sum')
        self.num_classes = num_classes

    def update(self, preds, target):
        max_id = torch.argmax(preds, axis=-1)
        assert max_id.shape == target.shape

        for cls in range(self.num_classes):
            pred_mask = (max_id == cls)
            target_mask = (target == cls)
            tp_mask = pred_mask & target_mask

            tp = tp_mask.sum()
            fp = pred_mask.sum() - tp
            fn = target_mask.sum() - tp
            self.TP[cls] += tp
            self.FP[cls] += fp
            self.FN[cls] += fn

    def compute(self, average='micro', eps=1e-6):
        if average == 'macro':
            precisions = self.TP.float() / (self.TP.float() + self.FP.float() + eps)
            recalls = self.TP.float() / (self.TP.float() + self.FN.float() + eps)
            f1 = (2 * (precisions * recalls) / (precisions + recalls + eps)).mean()
            precision = precisions.mean()
            recall = recalls.mean()
        elif average == 'micro':
            TP = self.TP.sum()
            FP = self.FP.sum()
            FN = self.FN.sum()
            precision = TP / (TP + FP + eps)
            recall = TP / (TP + FN + eps)
            f1 = 2 * (precision * recall) / (precision + recall)

        return f1, precision, recall