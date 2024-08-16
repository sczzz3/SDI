
import torch
import torch.nn as nn
import torch.nn.functional as F
        

class PairMarginRankLoss(nn.Module):
    def __init__(self, pair_margins=None):
        """
        Initialize the loss function with a mapping of pair types to margins.
        
        :param pair_margins: A dictionary or any mapping where each key is a tuple representing a pair
                             (or pair type) and the value is the margin for that pair.
        """
        super(PairMarginRankLoss, self).__init__()
        # Store the pair margins
        pair_margins = {
            (0, 1): 1, (1, 0): 1,
            (0, 2): 1.5, (2, 0): 1.5,
            (0, 3): 3, (3, 0): 3,
            (0, 4): 1.2, (4, 0): 1.2,

            (1, 2): 0.5, (2, 1): 0.5,
            (1, 3): 2, (3, 1): 2,

            (2, 3): 1.5, (3, 2): 1.5,

        }
        self.pair_margins = pair_margins

    def forward(self, pred_depth, label):

        pred_depth = torch.combinations(pred_depth.squeeze(dim=1))
        label = torch.combinations(label)

        margins = torch.tensor([self.pair_margins.get(tuple(pair), 1.0) for pair in label.tolist()])
        margins = margins.to(pred_depth.device, non_blocking=True)
        
        # Exclude the relations between REM and the other sleep stagess
        uncertain_relationships = [[1, 4], [4, 1], [2, 4], [4, 2], [3, 4], [4, 3]]
        uncertain_relationships_tensor = torch.tensor(uncertain_relationships).to(pred_depth.device, non_blocking=True)
        label_expanded = label.unsqueeze(1).expand(-1, uncertain_relationships_tensor.size(0), -1)
        is_uncertain = torch.any(torch.all(label_expanded == uncertain_relationships_tensor, dim=2), dim=1)
        
        penalties = torch.relu(margins + (pred_depth[:, 1] - pred_depth[:, 0])) * (label[:, 0] > label[:, 1])
        penalties += torch.relu(margins + (pred_depth[:, 0] - pred_depth[:, 1])) * (label[:, 0] < label[:, 1])

        loss = torch.mean(penalties * (~is_uncertain))
        return loss
