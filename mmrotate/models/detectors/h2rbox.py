# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict

from mmdet.models.detectors.single_stage import SingleStageDetector
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from torch import Tensor

from mmrotate.registry import MODELS


@MODELS.register_module()
class H2RBoxDetector(SingleStageDetector):
    """Implementation of `H2RBox <https://arxiv.org/abs/2210.06742>`_"""

    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

    def loss(self, multi_batch_inputs: Dict[str, Tensor],
             multi_batch_data_samples: Dict[str, SampleList]) -> dict:
        """Calculate losses from multi-branch inputs and data samples.

        Args:
            multi_batch_inputs (Dict[str, Tensor]): The dict of multi-branch
                input images, each value with shape (N, C, H, W).
                Each value should usually be mean centered and std scaled.
            multi_batch_data_samples (Dict[str, List[:obj:`DetDataSample`]]):
                The dict of multi-branch data samples.
        Returns:
            dict: A dictionary of loss components
        """
        x1 = self.extract_feat(multi_batch_inputs['a'])
        x2 = self.extract_feat(multi_batch_inputs['b'])
        losses = self.bbox_head.loss(x1, x2, multi_batch_data_samples['a'],
                                     multi_batch_data_samples['b'])
        return losses
