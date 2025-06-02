_base_ = ['./roi_trans_r50_fpn_1x_kari_objects_33cls_mini_le90.py']

data_root = 'data/split_ms_kari_objects_33cls_mini/'
data = dict(
    train=dict(
        ann_file=data_root + 'train/annfiles/',
        img_prefix=data_root + 'train/images/'),
    val=dict(
        ann_file=data_root + 'val/annfiles/',
        img_prefix=data_root + 'val/images/'),
    test=dict(
        ann_file=data_root + 'val/annfiles/',
        img_prefix=data_root + 'val/images/'))
