import torch
import torch.nn.functional as F

# Define target class indices and predicted log probabilities
target = torch.tensor([1, 0, 2])
log_probs = torch.tensor([[-0.3, -2.5, -1.0],
                          [-2.0, -0.1, -3.0],
                          [-1.5, -2.0, -0.8]])

# Apply log softmax to the predicted log probabilities
log_probs = F.log_softmax(log_probs, dim=1)

# Compute the negative log likelihood loss
loss = F.nll_loss(log_probs, target)

print(loss.item())