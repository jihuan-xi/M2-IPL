import cv2
import torch
import numpy as np
from torchvision import ops
import torch.nn.functional as F


class PromptMemoryBank:
    def __init__(self, cap=5, threshold=0.6):
        self.capacity = cap
        self.threshold = threshold
        self.rgb_buffer = []
        self.tir_buffer = []

    def update(self, rgb, tir, score):
        if self.threshold > score:
            return

        if rgb:
            rgb_copy = rgb.detach().clone() if isinstance(rgb, torch.Tensor) \
                else [p.detach().clone() for p in rgb]
            self.rgb_buffer.append((rgb_copy, score))

        if tir:
            tir_copy = tir.detach().clone() if isinstance(tir, torch.Tensor) \
                else [p.detach().clone() for p in tir]
            self.tir_buffer.append((tir_copy, score))

        if len(self.rgb_buffer) > self.capacity:
            self.rgb_buffer.pop(0)
        if len(self.tir_buffer) > self.capacity:
            self.tir_buffer.pop(0)

    def aggregate(self):
        # return

        tir, rgb = None, None
        if len(self.rgb_buffer):
            rgb = buffer_aggregate(self.rgb_buffer)
        if len(self.tir_buffer):
            tir = buffer_aggregate(self.tir_buffer)
        return rgb, tir


def buffer_aggregate(buffer):
    if not buffer:
        return None
    scores = torch.tensor([item[1] for item in buffer])
    weights = scores / (scores.sum() + 1e-6)
    if isinstance(buffer[0][0], list):
        res = []
        for i in range(len(buffer[0][0])):
            stack_prompt = torch.stack([b[0][i] for b in buffer])
            weight = weights.to(stack_prompt.device).view(-1, 1, 1, 1)
            res.append((stack_prompt * weight).sum(dim=0))
        return torch.stack([r for r in res])
    stack_prompt = torch.stack([b[0] for b in buffer])
    weight = weights.to(stack_prompt.device).view(-1, 1, 1, 1)
    return (stack_prompt * weight).sum(dim=0)


class FeatureValidator:
    def __init__(self):
        self.history_rgb_feat = None
        self.history_tir_feat = None
        self.sim_threshold = 0.75

    def extract_feature(self, feat_map, pred_bbox, search_feat_size=16):
        search_tokens_len = search_feat_size * search_feat_size
        search_feat = feat_map[:, -search_tokens_len:, :]
        spacial_feat = search_feat.transpose(1, 2).reshape(1, -1, search_feat_size, search_feat_size)

        cx, cy, w, h = pred_bbox
        x1 = (cx - w / 2) * search_feat_size
        y1 = (cy - h / 2) * search_feat_size
        x2 = (cx + w / 2) * search_feat_size
        y2 = (cy + h / 2) * search_feat_size

        rois = torch.tensor([[0, x1, y1, x2, y2]], dtype=torch.float32, device=spacial_feat.device)
        target = ops.roi_align(spacial_feat, rois, output_size=(1, 1), spatial_scale=1.0)
        return target.view(-1)

    def update_history_feature(self, feat_map, pred_bbox, score_map_max, input_state='rgb'):
        if score_map_max < 0.85:
            return
        if input_state == 'rgb':
            rgb_feat = self.extract_feature(feat_map, pred_bbox)
            self.history_rgb_feat = rgb_feat.detach().clone()
        elif input_state == 'tir':
            tir_feat = self.extract_feature(feat_map, pred_bbox)
            self.history_tir_feat = tir_feat.detach().clone()

    def varify_proposal(self, feat_map, pred_bbox, input_state='rgbtir'):
        sim_scores = []
        if input_state in ['rgbtir', 'rgb'] and self.history_rgb_feat is not None:
            rgb_feat = self.extract_feature(feat_map[0], pred_bbox)
            sim_score = F.cosine_similarity(rgb_feat.unsqueeze(0), self.history_rgb_feat.unsqueeze(0))
            sim_scores.append(sim_score.item())
        if input_state in ['rgbtir', 'tir'] and self.history_tir_feat is not None:
            tir_feat = self.extract_feature(feat_map[1], pred_bbox)
            sim_score = F.cosine_similarity(tir_feat.unsqueeze(0), self.history_tir_feat.unsqueeze(0))
            sim_scores.append(sim_score.item())

        if len(sim_scores) == 0:
            return True, 0, [0, 0]

        final_sim = sum(sim_scores) / len(sim_scores)
        return final_sim >= self.sim_threshold, final_sim, sim_scores


class KalmanFilter:
    def __init__(self, bbox):
        assert bbox is not None

        self.kf = cv2.KalmanFilter(8, 4, 0)
        self.kf.transitionMatrix = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.kf.transitionMatrix[i, i + 4] = 1.0

        self.kf.measurementMatrix = np.eye(4, 8, dtype=np.float32)

        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 0.01
        self.kf.processNoiseCov[4:, 4:] *= 0.8

        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32)

        self.kf.errorCovPost = np.eye(8, dtype=np.float32)
        self.kf.errorCovPost[:4, :4] *= 10.0
        self.kf.errorCovPost[4:, 4:] *= 1000.0

        # state = np.zeros((8, 1), dtype=np.float32)
        # state[:4, 0] = bbox
        # self.kf.statePost = state
        # self.kf.statePre = state

        self.kf.statePost = np.zeros((8, 1), dtype=np.float32)
        cx = bbox[0] + bbox[2] / 2
        cy = bbox[1] + bbox[3] / 2
        # self.kf.statePost[:4] = np.array(bbox, dtype=np.float32).reshape(4, 1)
        self.kf.statePost[:4] = np.array([cx, cy, bbox[2], bbox[3]], dtype=np.float32).reshape(4, 1)
        self.kf.statePre = self.kf.statePost.copy()

    def update(self, bbox, score):
        # measurement = np.array(bbox, dtype=np.float32).reshape(4, 1)
        # if score is not None:
        #     adaptive_R = np.eye(4, dtype=np.float32) * (1.0 - score) * 10.0
        #     self.kf.measurementNoiseCov = adaptive_R + 1e-3
        
        # if score is None or score < 0.4:
        #     return
        # if score is not None:
        #     adaptive_R = np.eye(4, dtype=np.float32) * (1.0 - score) * 10.0
        #     self.kf.measurementNoiseCov = adaptive_R + 1e-3
        #     beta = 10.0
        #     scale_factor = np.exp(beta * (1.0 - score))
        #     scale_factor = max(scale_factor, 1000.0)
        #     self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * scale_factor

        cx = bbox[0] + bbox[2] / 2
        cy = bbox[1] + bbox[3] / 2
        measurement = np.array([cx, cy, bbox[2], bbox[3]], dtype=np.float32).reshape(4, 1)
        self.kf.correct(measurement)

    def predict(self):
        # pred = self.kf.predict()
        # pred_bbox = pred[:4].flatten().tolist()
        #
        # return [pred_bbox[0], pred_bbox[1], max(5, pred_bbox[2]), max(5, pred_bbox[3])]
        cx, cy, cw, ch = self.kf.predict().flatten().tolist()[:4]
        w = float(max(3.0, cw))
        h = float(max(3.0, ch))
        x = float(cx)
        y = float(cy)

        return [float(x - w / 2), float(y - h / 2), w, h]

    def predict_with_uncertainty(self):
        pred_bbox = self.predict()

        p_pre = self.kf.errorCovPre
        uncertainty = p_pre[0, 0] + p_pre[1, 1]
        return pred_bbox, uncertainty

    def get_state(self):
        return self.kf.statePost[:4, 0]
