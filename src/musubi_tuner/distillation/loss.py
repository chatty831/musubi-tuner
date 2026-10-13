import torch
import torch.nn.functional as F

def distillation_loss(student_pred: torch.Tensor, teacher_denoised: torch.Tensor, loss_type: str = "mse") -> torch.Tensor:
    """Compute distillation loss between student prediction and teacher's denoised output."""
    if loss_type == "mse":
        return F.mse_loss(student_pred, teacher_denoised)
    elif loss_type == "l1":
        return F.l1_loss(student_pred, teacher_denoised)
    elif loss_type == "huber":
        return F.smooth_l1_loss(student_pred, teacher_denoised)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
