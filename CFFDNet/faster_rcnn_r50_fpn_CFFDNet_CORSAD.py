model = dict(
    type='Faster_RCNN_CFFDNet',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    dfsc=True,
    gwf=True,
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 1.5],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)))
dataset_type = 'CORSAD'
data_root = 'D:/DL/CFFDNet/datasets/CORSAD/'
img_norm_cfg = dict(
    mean_list=([125.78, 123.14, 118.76], [31.93, 31.95, 32.08]),
    std_list=([51.86, 50.01, 49.82], [28.75, 28.89, 28.81]),
    to_rgb=True)
train_pipeline = [
    dict(
        type='LoadImagePairFromFile',
        modalities=('optical', 'sar')),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(640, 640), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='MultiNormalize',
        mean_list=([125.78, 123.14, 118.76], [31.93, 31.95, 32.08]),
        std_list=([51.86, 50.01, 49.82], [28.75, 28.89, 28.81]),
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels'],
        meta_keys=('filename', 'ori_filename', 'ori_shape', 'img_shape',
                   'pad_shape', 'scale_factor', 'flip', 'flip_direction',
                   'img_norm_cfg'))
]
test_pipeline = [
    dict(type='LoadImagePairFromFile',
         modalities=('optical', 'sar')),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 640),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='MultiNormalize',
                mean_list=([125.78, 123.14, 118.76], [31.93, 31.95, 32.08]),
                std_list=([51.86, 50.01, 49.82], [28.75, 28.89, 28.81]),
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='CORSAD',
        ann_file='D:/DL/CFFDNet/datasets/CORSAD/annotations/train2017.json',
        img_prefix='D:/DL/CFFDNet/datasets/CORSAD/train/',
        pipeline=[
            dict(
                type='LoadImagePairFromFile',
                modalities=('optical', 'sar')),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', img_scale=(640, 640), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='MultiNormalize',
                mean_list=([125.78, 123.14, 118.76], [31.93, 31.95, 32.08]),
                std_list=([51.86, 50.01, 49.82], [28.75, 28.89, 28.81]),
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['img', 'gt_bboxes', 'gt_labels'],
                meta_keys=('filename', 'ori_filename', 'ori_shape',
                           'img_shape', 'pad_shape', 'scale_factor', 'flip',
                           'flip_direction', 'img_norm_cfg'))
        ]),
    val=dict(
        type='CORSAD',
        ann_file='D:/DL/CFFDNet/datasets/CORSAD/annotations/val2017.json',
        img_prefix='D:/DL/CFFDNet/datasets/CORSAD/val/',
        pipeline=[
            dict(
                type='LoadImagePairFromFile',
                modalities=('optical', 'sar')),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(640, 640),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='MultiNormalize',
                        mean_list=([125.78, 123.14, 118.76], [31.93, 31.95, 32.08]),
                        std_list=([51.86, 50.01, 49.82], [28.75, 28.89, 28.81]),
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CORSAD',
        ann_file='D:/DL/CFFDNet/datasets/CORSAD/annotations/val2017.json',
        img_prefix='D:/DL/CFFDNet/datasets/CORSAD/val',
        pipeline=[
            dict(
                type='LoadImagePairFromFile',
                modalities=('optical', 'sar')),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(640, 640),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='MultiNormalize',
                        mean_list=([125.78, 123.14, 118.76], [31.93, 31.95, 32.08]),
                        std_list=([51.86, 50.01, 49.82], [28.75, 28.89, 28.81]),
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
evaluation = dict(interval=1, metric='bbox')
optimizer = dict(
    type='SGD', lr=0.0002, betas=(0.9, 0.999), weight_decay=0.05)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)
checkpoint_config = dict(interval=6)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
work_dir = 'D:/DL/CFFDNet/tools/work_dirs/faster_rcnn_r50_fpn_CFFDNet_CORSAD'
auto_resume = False
gpu_ids = [0]
