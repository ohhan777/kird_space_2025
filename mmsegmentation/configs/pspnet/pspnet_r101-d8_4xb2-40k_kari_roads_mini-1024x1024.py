_base_ = [
    '../_base_/models/pspnet_r50-d8_kari_roads_mini.py', '../_base_/datasets/kari_roads_mini_datasets.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
crop_size = (1024, 1024)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor)
model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))

