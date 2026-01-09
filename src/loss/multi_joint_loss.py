import math
import torch
import torch.nn as nn


class MultiJointLoss(nn.Module):
    def __init__(self, alpha: float):
        """Joint Domain and Classification Loss for two heads.

        Args:
            alpha: Ratio between domain loss and classification loss.
                   Final loss: alpha * domain_loss + (1 - alpha) * class_loss
        """
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.alpha = alpha
        self.neg_alpha = 1. - alpha

    @staticmethod
    def proportional(pos_value: torch.Tensor,
                     neg_value: torch.Tensor,
                     val: float, neg_val: float) -> torch.Tensor:
        """Combines the values using alpha to provide proper weighting."""
        return (val * pos_value) + (neg_val * neg_value)

    def forward(self,
                target_label: torch.Tensor,
                null_class_pred: torch.Tensor,
                embed_class_pred: torch.Tensor | None,
                null_domain_pred: torch.Tensor,
                embed_domain_pred: torch.Tensor | None,
                domain_label: torch.Tensor) -> dict[str, torch.Tensor]:
        """Calculate joint loss with classification and domain prediction."""
        # Domain prediction losses
        null_domain_loss = self.ce(null_domain_pred, domain_label)
        if embed_domain_pred is not None:
            embed_domain_loss = self.ce(embed_domain_pred, domain_label)
            domain_loss = 0.5 * (null_domain_loss + embed_domain_loss)
        else:
            embed_domain_loss = None
            domain_loss = null_domain_loss

        # Classification losses
        null_class_loss = self.ce(null_class_pred, target_label)
        if embed_class_pred is not None:
            embed_class_loss = self.ce(embed_class_pred, target_label)
            class_loss = 0.5 * (null_class_loss + embed_class_loss)
        else:
            embed_class_loss = None
            class_loss = null_class_loss

        # Combined joint losses
        null_joint_loss = self.proportional(null_domain_loss, null_class_loss,
                                            self.alpha, self.neg_alpha)
        if embed_class_pred is not None and embed_domain_pred is not None:
            embed_joint_loss = self.proportional(embed_domain_loss,
                                                 embed_class_loss,
                                                 self.alpha, self.neg_alpha)
            joint_loss = 0.5 * (null_joint_loss + embed_joint_loss)
        else:
            embed_joint_loss = None
            joint_loss = null_joint_loss

        return {
            "domain_loss": domain_loss,
            "class_loss": class_loss,
            "joint_loss": joint_loss,
            "null_joint_loss": null_joint_loss,
            "embed_joint_loss": embed_joint_loss
        }
