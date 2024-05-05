_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/tless_random_detection_p_1.0_optim_aug.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py',
    './retinanet_tta.py'
]

# optimizer
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))
