_base_ = './faster-rcnn_r50_fpn_1x_tless_random_texture.py'
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))
