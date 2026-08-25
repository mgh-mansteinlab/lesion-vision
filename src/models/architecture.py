"""SMP segmentation network: Unet or Unet++ depending on encoder support."""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

# timm ConvNeXt / MiT encoders are not compatible with SMP's Unet++ decoder.
_UNETPP_INCOMPATIBLE_PREFIXES = ("tu-convnext", "tu-convnextv2", "mit_b")


def uses_plain_unet(backbone: str) -> bool:
    return any(backbone.startswith(prefix) for prefix in _UNETPP_INCOMPATIBLE_PREFIXES)


class SMPSegmentationNet(nn.Module):
    """segmentation-models-pytorch Unet / Unet++ wrapper.

    Default backbone ``tu-convnext_base`` uses ``smp.Unet``. Compatible
    encoders such as EfficientNet and ResNet use ``smp.UnetPlusPlus``.
    """

    def __init__(
        self,
        n_classes=4,
        backbone="tu-convnext_base",
        encoder_weights="imagenet",
        activation_checkpointing=False,
    ):
        super().__init__()
        self.backbone = backbone
        self._activation_checkpointing = activation_checkpointing
        arch_cls = smp.Unet if uses_plain_unet(backbone) else smp.UnetPlusPlus
        self.model = arch_cls(
            encoder_name=backbone,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=n_classes,
            activation=None,
        )

    def forward(self, x):
        if self._activation_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(self.model, x, use_reentrant=False)
        return self.model(x)
