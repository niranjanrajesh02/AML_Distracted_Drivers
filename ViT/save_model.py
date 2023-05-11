import torch
import torchvision
from model import VisionTransformer

vit = VisionTransformer(img_size=32, patch_size=2, n_classes=10, depth=6, n_heads=6, p=0.1, attn_p=0.1)

torch.save(vit, "./ViT_model.pth")
