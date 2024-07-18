_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn_keypose.py',
    '../_base_/datasets/keypose_random_texture_instance.py',
    '../_base_/schedules/schedule_30.py', '../_base_/default_runtime.py'
]
