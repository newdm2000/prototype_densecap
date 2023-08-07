from torch import nn
import torch.nn.functional as F

from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

class VisualEncoder(GeneralizedRCNN):
    