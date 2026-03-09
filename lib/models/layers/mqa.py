import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

VALID_STATUS = 'VALID'
MISSING_STATUS = 'MISSING'
SCORE_BAD = 0.4
COS_FROZEN = 0.9
COS_MOTION = 0.6


class ModalityQualityAssessment(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.linear = nn.Linear(128, 1)

    def forward(self, x):
        feature = self.conv(x).flatten(1)
        score = torch.sigmoid(self.linear(feature))
        return feature, score


class MQAModule:
    def __init__(self, device='cuda', checkpoint=None):
        self.mqa = ModalityQualityAssessment()
        self.buffer_feature = {'rgb': None, 'tir': None}
        if checkpoint:
            self.mqa.load_state_dict(torch.load(checkpoint, map_location='cpu'), strict=True)
        self.mqa.eval()

    def forward(self, rgb, tir):
        W, H, _ = rgb.shape
        rgb = torch.Tensor(rgb).reshape(1, 3, W, H)
        tir = torch.Tensor(tir).reshape(1, 3, W, H)
        rgb_sm = F.interpolate(rgb, size=(64, 64), mode='bilinear')
        tir_sm = F.interpolate(tir, size=(64, 64), mode='bilinear')

        rgb_status = self._assess_img(rgb_sm, 'rgb')
        tir_status = self._assess_img(tir_sm, 'tir')
        if rgb_status == MISSING_STATUS and tir_status == MISSING_STATUS:
            return 'skip'
        if rgb_status == MISSING_STATUS:
            return 'tir'
        if tir_status == MISSING_STATUS:
            return 'rgb'
        return 'rgbtir'

    def _assess_img(self, img, img_type):
        curr_feature, curr_score = self.mqa(img)
        curr_feature = F.normalize(curr_feature, p=2, dim=1)
        status = VALID_STATUS

        if self.buffer_feature[img_type] is None:
            if curr_score < SCORE_BAD:
                status = MISSING_STATUS
            self.buffer_feature[img_type] = curr_feature.detach()
            return status

        pre_feature = self.buffer_feature[img_type]
        cos_sim = (curr_feature * pre_feature).sum(dim=1)
        if curr_score < SCORE_BAD or cos_sim > COS_FROZEN:
            status = MISSING_STATUS
        self.buffer_feature[img_type] = curr_feature.detach()
        return status

    def reset_buffer(self):
        self.buffer_feature = {'rgb': None, 'tir': None}


def get_input_state(image_v, image_i):
    assert image_v is not None and image_i is not None
    input_state = ''
    input_state += '' if is_black(image_v) else 'rgb'
    input_state += '' if is_black(image_i) else 'tir'
    return input_state if input_state else 'skip'


def is_black(image):
    assert image is not None
    if isinstance(image, str):
        image = cv2.imread(image)
    return np.mean(image) < 255 * 0.1
