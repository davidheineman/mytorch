import torch

EPS = 1e-7


def softmax(x):
    """
    σ(x) = e^z_i / Σ_j e^z_j
    """
    exp_x = torch.exp(x - torch.max(x, dim=-1, keepdim=True)[0]) # The second term prevents numerical overflow
    return exp_x / torch.sum(exp_x, dim=-1, keepdim=True)


def log_softmax(x):
    return torch.log(torch.softmax(x))


def nll_loss(y_true, y_pred):
    """
    Negative Log Likelihood Loss
    ℒ(y, p) = -Σ(log(p_i))
    """
    target_probs = y_pred[torch.arange(y_true), y_true]
    loss = -torch.mean(torch.log(target_probs))
    return loss


def cross_entropy(y_true, y_pred):
    """
    Cross Entropy Loss. Equivalent to nll_loss(log(softmax(logits)), targets)
    ℒ(y, p) = -Σ(y_i * log(p_i))
    """
    softmax_probs = softmax(y_pred)
    target_probs = softmax_probs[torch.arange(y_true), y_true]
    loss = -torch.mean(torch.log(target_probs))
    return loss


def bce(y_true, y_pred):
    """
    Binary Cross Entropy Loss
    ℒ(y, p) = -Σ(y*log(p) + (1-y)*log(1-p))
    """
    loss = -(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))
    loss = torch.mean(loss)
    return loss


def mse_loss(y_true, y_pred):
    """
    Mean Squared Error
    ℒ(y, p) = Σ(yt - yp)^2
    """
    squared_error = (y_true - y_pred)**2
    loss = torch.mean(squared_error)
    return loss


def kl_div():
    raise NotImplementedError()


def sigmoid(x):
    """
    S(x) = 1/1+e^-x
    """
    return 1 / (1 + torch.exp(-x))


def sigmoid_stable(x):
    """
    Numerically stable sigmoid, calculate positive/negatives differently to avoid underflow
    
    σ(x) = 1/1+e^-x if x >=0 else e^x/1+e^x
    """
    pos_mask, neg_mask = (x >= 0), (x < 0)
    pos_output, neg_output = torch.zeros_like(x), torch.zeros_like(x)
    
    # σ(x) = 1/1+e^-x  (if x >=0)
    pos_output[pos_mask] = 1 / (1 + torch.exp(-x[pos_mask]))
    
    # σ(x) = e^x/1+e^x (if x < 0)
    neg_output[neg_mask] = torch.exp(x[neg_mask]) / (torch.exp(x[neg_mask]) + 1)
    
    return pos_output + neg_output


def logsigmoid(x):
    return torch.log(sigmoid_stable(x))


def tanh(x):
    """
    tanh(x) = (e^x - e^-x) / (e^x + e^-x)
    """
    return torch.tanh(x)


def accuracy(y_true, y_pred):
    """
    acc = (TP + TN) / (TP + TN + FP + FN) = T / (T + F)
    """
    acc = (y_true == y_pred).sum() / len(y_true)
    return acc


def f1(y_true, y_pred):
    """
    f1 = 2 * (p * r) / (p + r)
    p  = TP / (TP + FP)
    r  = TP / (TP + FN)
    """
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()

    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)

    f1 = 2 * (precision * recall) / (precision + recall + EPS)

    return f1


def cosine_similarity():
    raise NotImplementedError()


def preference_loss(policy_chosen_logps: torch.FloatTensor,
                    policy_rejected_logps: torch.FloatTensor,
                    reference_chosen_logps: torch.FloatTensor,
                    reference_rejected_logps: torch.FloatTensor,
                    beta: float,
                    label_smoothing: float = 0.0,
                    ipo: bool = False,
                    reference_free: bool = False):
    """Compute the DPO loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
        label_smoothing: conservativeness for DPO loss, which assumes that preferences are noisy (flipped with probability label_smoothing)
        ipo: If True, use the IPO loss instead of the DPO loss.
        reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the DPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    """
    pi_logratios  = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    if reference_free: ref_logratios = 0

    logits = pi_logratios - ref_logratios  # also known as h_{\pi_\theta}^{y_w,y_l}

    if ipo:
        losses = (logits - 1/(2 * beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
    else:
        # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
        losses = -logsigmoid(beta * logits) * (1 - label_smoothing) - logsigmoid(-beta * logits) * label_smoothing

    chosen_rewards   = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    return losses, chosen_rewards, rejected_rewards