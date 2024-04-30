from torchmetrics import Metric
import torch

class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target):
        # [TODO] The preds (B x C tensor), so take argmax to get index with highest confidence
        max_id = torch.argmax(preds, axis=-1)
        assert max_id.shape == target.shape

        correct = torch.count_nonzero(max_id==target)

        # [TODO] check if preds and target have equal shape


        # [TODO] Cound the number of correct prediction


        # Accumulate to self.correct
        self.correct += correct

        # Count the number of elements in target
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
        # [TODO] The preds (B x C tensor), so take argmax to get index with highest confidence
        max_id = torch.argmax(preds, axis=-1)
        assert max_id.shape == target.shape

        for cls in range(self.num_classes):
            pred_mask = (max_id == cls)
            target_mask = (target == cls)
            tp_mask = torch.logical_and(pred_mask, target_mask)

            tp = torch.count_nonzero(tp_mask)
            fp = torch.count_nonzero(pred_mask) - tp
            fn = torch.count_nonzero(target_mask) - tp
            self.TP[cls] += tp
            self.FP[cls] += fp
            self.FN[cls] += fn

    def compute(self, eps=1e-6):
        precisions = self.TP.float() / (self.TP.float() + self.FP.float() + eps)
        recalls = self.TP.float() / (self.TP.float() + self.FN.float() + eps)
        return torch.mean(2 * (precisions * recalls) / (precisions + recalls + eps)), torch.mean(precisions), torch.mean(recalls)