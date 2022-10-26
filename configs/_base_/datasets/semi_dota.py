# dataset settings
dataset_type = 'CocoRboxDataset'
data_root = 'data/split_ss_dota/'
file_client_args = dict(backend='disk')

color_space = [
    [dict(type='mmdet.ColorTransform')],
    [dict(type='mmdet.AutoContrast')],
    [dict(type='mmdet.Equalize')],
    [dict(type='mmdet.Sharpness')],
    [dict(type='mmdet.Posterize')],
    [dict(type='mmdet.Solarize')],
    [dict(type='mmdet.Color')],
    [dict(type='mmdet.Contrast')],
    [dict(type='mmdet.Brightness')],
]

geometric = [
    [dict(type='mmdet.Rotate')],
    [dict(type='mmdet.ShearX')],
    [dict(type='mmdet.ShearY')],
    [dict(type='mmdet.TranslateX')],
    [dict(type='mmdet.TranslateY')],
]

scale = [(1333, 768), (1333, 1280)]

branch_field = ['sup', 'unsup_teacher', 'unsup_student']
# pipeline used to augment labeled data,
# which will be sent to student model for supervised training.
sup_pipeline = [
    dict(type='mmdet.LoadImageFromFile', file_client_args=file_client_args),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.RandomResize', scale=scale, keep_ratio=True, resize_type='mmdet.Resize'),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='mmdet.RandAugment', aug_space=color_space, aug_num=1),
    dict(type='mmdet.FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(
        type='mmdet.MultiBranch',
        branch_field=branch_field,
        sup=dict(type='mmdet.PackDetInputs'))
]

# pipeline used to augment unlabeled data weakly,
# which will be sent to teacher model for predicting pseudo instances.
weak_pipeline = [
    dict(type='mmdet.RandomResize', scale=scale, keep_ratio=True, resize_type='mmdet.Resize'),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction',
                   'homography_matrix')),
]

# pipeline used to augment unlabeled data strongly,
# which will be sent to student model for unsupervised training.
strong_pipeline = [
    dict(type='mmdet.RandomResize', scale=scale, keep_ratio=True, resize_type='mmdet.Resize'),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(
        type='mmdet.RandomOrder',
        transforms=[
            dict(type='mmdet.RandAugment', aug_space=color_space, aug_num=1),
            dict(type='mmdet.RandAugment', aug_space=geometric, aug_num=1),
        ]),
    dict(type='mmdet.FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction',
                   'homography_matrix')),
]

# pipeline used to augment unlabeled data into different views
unsup_pipeline = [
    dict(type='mmdet.LoadImageFromFile', file_client_args=file_client_args),
    dict(type='mmdet.LoadEmptyAnnotations'),
    dict(
        type='mmdet.MultiBranch',
        branch_field=branch_field,
        unsup_teacher=weak_pipeline,
        unsup_student=strong_pipeline,
    )
]

test_pipeline = [
    dict(type='mmdet.LoadImageFromFile', file_client_args=file_client_args),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    # avoid bboxes being resized
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

batch_size = 2
num_workers = 2
# There are two common semi-supervised learning settings on the dota dataset：
# (1) Divide the train2017 into labeled and unlabeled datasets
# by a fixed percentage, such as 1%, 2%, 5% and 10%.
# The format of labeled_ann_file and unlabeled_ann_file are
# train.{fold}@{percent}.json, and train.{fold}@{percent}-unlabeled.json
# `fold` is used for cross-validation, and `percent` represents
# the proportion of labeled data in the train.
# (2) Choose the dota as the labeled dataset
# and DIOR as the unlabeled dataset.
# The labeled_ann_file and unlabeled_ann_file are
# We use (1) configuration by default.
labeled_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='train/semi_anns/train.1@1.json',
    data_prefix=dict(img='train/images/'),
    filter_cfg=dict(filter_empty_gt=True),
    pipeline=sup_pipeline)

unlabeled_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='train/semi_anns/train.1@1-unlabeled.json',
    data_prefix=dict(img='train/images/'),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=unsup_pipeline)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(
        type='mmdet.GroupMultiSourceSampler',
        batch_size=batch_size,
        source_ratio=[1, 1]),
    dataset=dict(
        type='mmdet.ConcatDataset', datasets=[labeled_dataset, unlabeled_dataset]))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='val/val.json',
        data_prefix=dict(img='val/images/'),
        test_mode=True,
        pipeline=test_pipeline))

test_dataloader = val_dataloader

val_evaluator = dict(type='DOTAMetric', metric='mAP')
test_evaluator = val_evaluator
