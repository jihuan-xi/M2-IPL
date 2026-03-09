import numpy as np
from lib.test.evaluation.data import Sequence, Sequence_RGBT, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import os


class UrvisDataset(BaseDataset):
    # LasHeR dataset
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.urvis_path
        # self.base_path = r'/data_G/urvis/public_data/'
        # self.base_path = r'/urvis/public_data/'
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_info):
        sequence_path = sequence_info['path']
        anno_path = sequence_info['anno_path']
        ground_truth_rect = load_text(str(anno_path), delimiter=[' ', '\t', ','], dtype=np.float64)
        ground_truth_rect = np.array([ground_truth_rect])
        # img_list_v = sorted([p for p in os.listdir(os.path.join(sequence_path, 'visible')) if
        #                      os.path.splitext(p)[1] in ['.jpg', '.png', '.bmp']])
        # frames_v = [os.path.join(sequence_path, 'visible', img) for img in img_list_v]
        #
        # img_list_i = sorted([p for p in os.listdir(os.path.join(sequence_path, 'infrared')) if
        #                      os.path.splitext(p)[1] in ['.jpg', '.png', '.bmp']])
        # frames_i = [os.path.join(sequence_path, 'infrared', img) for img in img_list_i]
        img_list_v = sorted([p for p in os.listdir(os.path.join(sequence_path, 'RGB')) if
                             os.path.splitext(p)[1] in ['.jpg', '.png', '.bmp']])
        frames_v = [os.path.join(sequence_path, 'RGB', img) for img in img_list_v]

        img_list_i = sorted([p for p in os.listdir(os.path.join(sequence_path, 'TIR')) if
                             os.path.splitext(p)[1] in ['.jpg', '.png', '.bmp']])
        frames_i = [os.path.join(sequence_path, 'TIR', img) for img in img_list_i]
        # Convert gt
        if ground_truth_rect.shape[1] > 4:
            gt_x_all = ground_truth_rect[:, [0, 2, 4, 6]]
            gt_y_all = ground_truth_rect[:, [1, 3, 5, 7]]

            x1 = np.amin(gt_x_all, 1).reshape(-1, 1)
            y1 = np.amin(gt_y_all, 1).reshape(-1, 1)
            x2 = np.amax(gt_x_all, 1).reshape(-1, 1)
            y2 = np.amax(gt_y_all, 1).reshape(-1, 1)

            ground_truth_rect = np.concatenate((x1, y1, x2 - x1, y2 - y1), 1)
        return Sequence_RGBT(sequence_info['name'], frames_v, frames_i, 'urvis', ground_truth_rect)

    def __len__(self):
        return len(self.sequence_info_list)

    def _get_sequence_list(self):  # LasHeR_better
        # sequence_list = ['DJI_20251111210203gtwo_uav_viewq', 'DJI_20251030103125gtwo_uav_viewq',
        #                  'DJI_20251119150546gtwo_uav_viewq', 'DJI_20251104155239gtwo_ground_viewq',
        #                  'DJI_20251104162144gtwo_ground_viewq', 'DJI_20251028204148gtwo_uav_viewq',
        #                  'DJI_20251030110359gtwo_ground_viewq', 'DJI_20251203153251gtwo_ground_viewq',
        #                  'DJI_20251104155202gtwo_ground_viewq', 'DJI_20251104155734gtwo_ground_viewq',
        #                  'DJI_20251118210719gtwo_ground_viewq', 'DJI_20251104154653gtwo_ground_viewq',
        #                  'DJI_20251023104854gtwo_uav_viewq', 'DJI_20251028205703gtwo_ground_viewq',
        #                  'DJI_20251030110820gtwo_uav_viewq', 'DJI_20251029153737gtwo_uav_viewq',
        #                  'DJI_20251104155318gtwo_ground_viewq', 'DJI_20251111202112gtwo_uav_viewq',
        #                  'DJI_20251030113716gtwo_uav_viewq', 'DJI_20251030103125gtwo_ground_viewq',
        #                  'DJI_20251216174631gtwo_uav_viewq', 'DJI_20251021201131gtwo_ground_viewq',
        #                  'DJI_20251111212655gtwo_uav_viewq', 'DJI_20251111202412gtwo_uav_viewq',
        #                  'DJI_20251030103636gtwo_ground_viewq', 'DJI_20251111204618gtwo_ground_viewq',
        #                  'DJI_20251203154353gtwo_ground_viewq', 'DJI_20251030110347gtwo_ground_viewq',
        #                  'DJI_20251023104702gtwo_ground_viewq', 'DJI_20251119150941gtwo_uav_viewq',
        #                  'DJI_20251029150055gtwo_uav_viewq', 'DJI_20251203153442gtwo_uav_viewq',
        #                  'DJI_20251111212847gtwo_ground_viewq', 'DJI_20251111201447gtwo_ground_viewq',
        #                  'DJI_20251111210036gtwo_ground_viewq', 'DJI_20251022160154gtwo_uav_viewq',
        #                  'DJI_20251104162220gtwo_ground_viewq', 'DJI_20251021201055gtwo_ground_viewq',
        #                  'DJI_20251022160604gtwo_ground_viewq', 'DJI_20251203153117gtwo_uav_viewq',
        #                  'DJI_20251029145522gtwo_uav_viewq', 'DJI_20251111201328gtwo_ground_viewq',
        #                  'DJI_20251203151801gtwo_ground_viewq', 'DJI_20251023104539gtwo_ground_viewq',
        #                  'DJI_20251030114355gtwo_ground_viewq', 'DJI_20251111205026gtwo_ground_viewq',
        #                  'DJI_20251104154537gtwo_ground_viewq', 'DJI_20251104154537gtwo_uav_viewq',
        #                  'DJI_20251028205624gtwo_ground_viewq', 'DJI_20251111205557gtwo_uav_viewq',
        #                  'DJI_20251111205710gtwo_ground_viewq', 'DJI_20251111213300gtwo_uav_viewq',
        #                  'DJI_20251203151102gtwo_ground_viewq', 'DJI_20251203154554gtwo_ground_viewq',
        #                  'DJI_20251104160124gtwo_uav_viewq', 'DJI_20251030104957gtwo_uav_viewq',
        #                  'DJI_20251203153706gtwo_uav_viewq', 'DJI_20251029153809gtwo_uav_viewq',
        #                  'DJI_20251028205024gtwo_ground_viewq', 'DJI_20251029150119gtwo_uav_viewq']

        sequence_list = os.listdir(self.base_path)

        sequence_info_list = []
        for i in range(len(sequence_list)):
            sequence_info = {}
            sequence_info["name"] = sequence_list[i]
            sequence_info["path"] = self.base_path + sequence_info["name"]
            # sequence_info["startFrame"] = int('1')
            # print(end_frame[i])
            # sequence_info["endFrame"] = end_frame[i]

            # sequence_info["nz"] = int('6')
            # sequence_info["ext"] = 'jpg'
            sequence_info["anno_path"] = sequence_info["path"] + '/init.txt'
            # sequence_info["object_class"] = 'person'
            sequence_info_list.append(sequence_info)
        return sequence_info_list
