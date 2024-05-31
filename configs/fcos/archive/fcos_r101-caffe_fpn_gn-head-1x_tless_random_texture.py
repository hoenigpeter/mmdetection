_base_ = './fcos_r50-caffe_fpn_gn-head_1x_tless_random_p_1.0.py'

# model settings
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron/resnet101_caffe')))
