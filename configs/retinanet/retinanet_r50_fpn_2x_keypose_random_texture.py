_base_ = [
    '../_base_/models/retinanet_r50_fpn_keypose.py',
    '../_base_/datasets/keypose_random_texture_detection.py',
    '../_base_/schedules/schedule_30.py', '../_base_/default_runtime.py',
    './retinanet_tta.py'
]

# optimizer
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))
