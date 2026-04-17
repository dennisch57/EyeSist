import os
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import cv2

from runtime_config import get_runtime_device
from training_config import LABELS, NUM_CLASSES, RETRAIN_EXPERIMENT_CONFIG

CLASS_TO_IDX = {c: i for i, c in enumerate(LABELS)}
IDX_TO_CLASS = {i: c for i, c in enumerate(LABELS)}

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "ethxgaze_backbone_20260415121958.pth")


# ── ETH-XGaze ResNet-50 (forward() returns 2048-d features, fc never called) ──

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        self.groups = 1
        self.base_width = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, 1000)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample,
                        self.groups, self.base_width, self.dilation, norm_layer)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x  # (B, 2048, 1, 1) — fc intentionally skipped


def resnet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


class GazeClassifier(nn.Module):
    def __init__(self, backbone, num_classes=6, head_dense_units=None,
                 dropout=0.3, batch_norm_in_head=True):
        super().__init__()
        if head_dense_units is None:
            head_dense_units = []
        self.backbone = backbone
        head_layers = []
        in_features = 2048
        for units in head_dense_units:
            head_layers.append(nn.Linear(in_features, units))
            if batch_norm_in_head:
                head_layers.append(nn.BatchNorm1d(units))
            head_layers.append(nn.ReLU(inplace=True))
            head_layers.append(nn.Dropout(dropout))
            in_features = units
        head_layers.append(nn.Linear(in_features, num_classes))
        self.classifier = nn.Sequential(*head_layers)

    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        return self.classifier(features)

    def extract_features(self, x):
        self.backbone.eval()
        with torch.no_grad():
            features = self.backbone(x)
            return features.view(features.size(0), -1)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False


# ── Transform (matches training: ResizeWithPad → ToTensor → ImageNet normalize) ──

class _ResizeWithPad:
    def __init__(self, target_size, fill=0):
        self.target_size = target_size
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        scale = self.target_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = transforms.functional.resize(img, (new_h, new_w))
        pad_left = (self.target_size - new_w) // 2
        pad_right = self.target_size - new_w - pad_left
        pad_top = (self.target_size - new_h) // 2
        pad_bottom = self.target_size - new_h - pad_top
        return transforms.functional.pad(img, [pad_left, pad_top, pad_right, pad_bottom], fill=self.fill)


TRANSFORM = transforms.Compose([
    _ResizeWithPad(224, fill=0),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ── Singleton model loader ──

_model: GazeClassifier | None = None


def get_model() -> GazeClassifier:
    global _model
    if _model is None:
        _model = load_model(MODEL_PATH)
    return _model


def load_model(path: str) -> GazeClassifier:
    device = get_runtime_device()
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    backbone = resnet50()
    model = GazeClassifier(
        backbone=backbone,
        num_classes=NUM_CLASSES,
        head_dense_units=RETRAIN_EXPERIMENT_CONFIG["head_dense_units"],
        dropout=RETRAIN_EXPERIMENT_CONFIG["dropout"],
        batch_norm_in_head=RETRAIN_EXPERIMENT_CONFIG["batch_norm_in_head"],
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def preprocess(img_bgr: np.ndarray) -> torch.Tensor:
    """BGR ndarray → (1, 3, 224, 224) tensor."""
    from PIL import Image
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    return TRANSFORM(pil).unsqueeze(0)


def extract_features(img_bgr: np.ndarray) -> np.ndarray:
    """Returns 2048-d backbone feature vector for a cropped eye image."""
    model = get_model()
    tensor = preprocess(img_bgr).to(get_runtime_device())
    return model.extract_features(tensor).squeeze().cpu().numpy()


def predict_base(img_bgr: np.ndarray) -> str:
    """Returns predicted class name using the base GazeClassifier (no Ridge)."""
    model = get_model()
    tensor = preprocess(img_bgr).to(get_runtime_device())
    with torch.no_grad():
        logits = model(tensor)
    idx = logits.argmax(dim=1).item()
    return IDX_TO_CLASS[idx]
