# Copyright (c) OpenMMLab. All rights reserved.
import copy

from mmrotate.registry import MODELS
from mmdet.structures import SampleList
from mmdet.models.detectors import SemiBaseDetector
from mmrotate.structures.bbox import RotatedBoxes

@MODELS.register_module()
class RotatedSemiBaseDetector(SemiBaseDetector):
    """Base class for semi-supervsed detectors.

    Semi-supervised detectors typically consisting of a teacher model
    updated by exponential moving average and a student model updated
    by gradient descent.

    Args:
        detector (:obj:`ConfigDict` or dict): The detector config.
        semi_train_cfg (:obj:`ConfigDict` or dict, optional):
            The semi-supervised training config.
        semi_test_cfg (:obj:`ConfigDict` or dict, optional):
            The semi-supervised testing config.
        data_preprocessor (:obj:`ConfigDict` or dict, optional): Config of
            :class:`DetDataPreprocessor` to process the input data.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def filter_pseudo_instances(self,
                                batch_data_samples: SampleList) -> SampleList:
        """Filter invalid pseudo instances from teacher model."""
        for data_samples in batch_data_samples:
            pseudo_bboxes = data_samples.gt_instances.bboxes
            if pseudo_bboxes.shape[0] > 0:
                w = pseudo_bboxes.widths
                h = pseudo_bboxes.heights
                data_samples.gt_instances = data_samples.gt_instances[
                    (w > self.semi_train_cfg.min_pseudo_bbox_wh[0])
                    & (h > self.semi_train_cfg.min_pseudo_bbox_wh[1])]
        return batch_data_samples

    def project_pseudo_instances(self, batch_pseudo_instances: SampleList,
                                 batch_data_samples: SampleList) -> SampleList:
        """Project pseudo instances."""
        for pseudo_instances, data_samples in zip(batch_pseudo_instances,
                                                  batch_data_samples):
            data_samples.gt_instances = copy.deepcopy(
                pseudo_instances.gt_instances)
            boxlist_bboxes = RotatedBoxes(data_samples.gt_instances.bboxes)
            boxlist_bboxes.project_(data_samples.homography_matrix)
            data_samples.gt_instances.bboxes = boxlist_bboxes
        return self.filter_pseudo_instances(batch_data_samples)
