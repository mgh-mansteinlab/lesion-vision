"""SMP segmentation network (Unet by default; other SMP decoders selectable)."""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

# timm ConvNeXt / MiT encoders are not compatible with SMP's Unet++ decoder.
_UNETPP_INCOMPATIBLE_PREFIXES = ("tu-convnext", "tu-convnextv2", "mit_b")


def uses_plain_unet(backbone: str) -> bool:
    return any(backbone.startswith(prefix) for prefix in _UNETPP_INCOMPATIBLE_PREFIXES)


# arch name -> SMP class. 'auto' keeps the original rule (Unet for ConvNeXt /
# MiT encoders, Unet++ otherwise), so checkpoints saved before the arch
# option existed rebuild exactly as trained.
ARCHS = {
    'unet': smp.Unet,
    'unetplusplus': smp.UnetPlusPlus,
    'deeplabv3plus': smp.DeepLabV3Plus,
    'upernet': smp.UPerNet,
    'segformer': smp.Segformer,
    'fpn': smp.FPN,
}


def resolve_arch(arch: str, backbone: str) -> str:
    if arch in (None, '', 'auto'):
        return 'unet' if uses_plain_unet(backbone) else 'unetplusplus'
    if arch not in ARCHS:
        raise ValueError(f"Unknown arch {arch!r}; choose from {sorted(ARCHS)} or 'auto'")
    return arch


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
        arch='auto',
    ):
        super().__init__()
        self.backbone = backbone
        self.arch = resolve_arch(arch, backbone)
        self._activation_checkpointing = activation_checkpointing
        arch_cls = ARCHS[self.arch]
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
