"""Paper MPMT ablation on COCO-C Gaussian severity 5.

This configuration enables TPT, deep VPT, TTWS, weak/strong Mean Teacher,
native Grounding DINO losses, and prompt-only EMA.  It contains no DINOv2,
IDM, Memory Enhancement, or Memory Hallucination path.
"""

import os


_base_ = '../grounding_dino_swin-t_pretrain_obj365.py'

detector = _base_.model
detector.text_prompt = True
detector.language_model.name = os.getenv(
    'TTAOD_BERT_PATH',
    '/bigdata/baoming/ttaod_foundation/download/bert-base-uncased')
detector.backbone = dict(
    **detector.backbone,
    prompt_type='prepend',
    num_tokens=10,
    prompt_deep=True,
    TTWS_init=False,
    TTWS_file=os.getenv(
        'TTAOD_TTWS_FILE',
        '/bigdata/baoming/TTAOD-F-paper-original-causal/evidence/ttws/'
        'prompt_coco_gaussian_noise.pth'))

model = dict(
    _delete_=True,
    type='TTAODSoftTeacher',
    detector=detector,
    data_preprocessor=dict(
        type='MultiBranchDataPreprocessor',
        data_preprocessor=detector.data_preprocessor),
    semi_train_cfg=dict(
        freeze_teacher=True,
        sup_weight=1.0,
        unsup_weight=1.0,
        pseudo_label_initial_score_thr=0.3,
        rpn_pseudo_thr=0.9,
        cls_pseudo_thr=0.3,
        reg_pseudo_thr=0.02,
        jitter_times=10,
        jitter_scale=0.06,
        min_pseudo_bbox_wh=(1e-2, 1e-2)),
    semi_test_cfg=dict(predict_on='teacher'))

dataset_type = 'CocoDataset'
data_root = os.getenv(
    'TTAOD_COCO_C_ROOT',
    '/bigdata/baoming/datasets/coco-c-q95-severity5-all15-20260818/')
ann_file = os.getenv(
    'TTAOD_GTFREE_ANN',
    '/bigdata/baoming/TTAOD-F-paper-original-causal/evidence/input/'
    'coco_val2017_images_categories_no_gt.json')
corruption_type = 'gaussian_noise/'

class_name = (
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush')
metainfo = dict(classes=class_name)
backend_args = None

color_space = [
    [dict(type='ColorTransform')],
    [dict(type='AutoContrast')],
    [dict(type='Equalize')],
    [dict(type='Sharpness')],
    [dict(type='Posterize')],
    [dict(type='Solarize')],
    [dict(type='Color')],
    [dict(type='Contrast')],
    [dict(type='Brightness')],
]

weak_pipeline = [
    dict(
        type='RandomChoice',
        transforms=[[
            dict(
                type='RandomChoiceResize',
                scales=[
                    (480, 1333), (512, 1333), (544, 1333), (576, 1333),
                    (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                    (736, 1333), (768, 1333), (800, 1333)],
                keep_ratio=True)]],),
    dict(
        type='PackDetInputs',
        meta_keys=(
            'img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor',
            'flip', 'flip_direction', 'text', 'custom_entities',
            'homography_matrix'))]

strong_pipeline = [
    dict(type='RandomResize', scale=[(1333, 400), (1333, 1200)],
         keep_ratio=True),
    dict(
        type='RandomOrder',
        transforms=[dict(
            type='RandAugment', aug_space=color_space, aug_num=1)]),
    dict(type='RandomErasing', n_patches=(1, 5), ratio=(0, 0.2)),
    dict(
        type='PackDetInputs',
        meta_keys=(
            'img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor',
            'flip', 'flip_direction', 'text', 'custom_entities',
            'homography_matrix'))]

runtime_test_pipeline = [
    dict(type='FixScaleResize', scale=(800, 1333), keep_ratio=True,
         backend='pillow'),
    dict(
        type='PackDetInputs',
        meta_keys=(
            'img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor',
            'text', 'custom_entities', 'tokens_positive',
            'homography_matrix'))]

unsup_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadEmptyAnnotations'),
    dict(
        type='MultiBranch',
        branch_field=['unsup_teacher', 'unsup_student', 'test'],
        unsup_teacher=weak_pipeline,
        unsup_student=strong_pipeline,
        test=runtime_test_pipeline)]

unlabeled_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    metainfo=metainfo,
    ann_file=ann_file,
    data_prefix=dict(img=corruption_type),
    return_classes=True,
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=unsup_pipeline,
    backend_args=backend_args)

train_dataloader = dict(
    _delete_=True,
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=unlabeled_dataset)

plain_test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    *runtime_test_pipeline]
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file=ann_file,
        data_prefix=dict(img=corruption_type),
        test_mode=True,
        return_classes=True,
        pipeline=plain_test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

# Built only for Runner compatibility. Evaluation is deliberately offline.
val_evaluator = dict(
    type='CocoMetric', ann_file=ann_file,
    metric='bbox', format_only=False, backend_args=backend_args)
test_evaluator = val_evaluator

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.2, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={
        'absolute_pos_embed': dict(decay_mult=0.0),
        # 0.2 * 0.1 = the paper's text-prompt LR 0.02.
        'student.tunable_linear': dict(lr_mult=0.1)}))

max_epochs = 1
param_scheduler = [dict(
    type='MultiStepLR', begin=0, end=max_epochs, by_epoch=True,
    milestones=[1], gamma=0.1)]

train_cfg = dict(
    type='PaperMPMTOnlineLoop',
    max_epochs=max_epochs,
    val_interval=1,
    artifact_dir='work_dirs/mpmt_no_memory_coco_gaussian/artifacts',
    max_batches=0)
val_cfg = dict(type='TeacherStudentValLoop')
test_cfg = dict(type='TestLoop')

custom_hooks = [dict(type='PaperPromptMeanTeacherHook', retention=0.999)]
default_hooks = dict(checkpoint=dict(interval=-1, save_last=False))
randomness = dict(seed=20261022, deterministic=False)

load_from = os.getenv(
    'TTAOD_CHECKPOINT',
    '/bigdata/baoming/TTAOD-F-paper-original-causal/download/'
    'grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_'
    '20231204_095047-b448804b.pth')
work_dir = 'work_dirs/mpmt_no_memory_coco_gaussian'
