from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset

@DATASETS.register_module()
class KariRoadsMiniDataset(BaseSegDataset):
    METAINFO = dict(
        classes=('background', 'motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'path', 'under construction',
                 'train guideway', 'airport runway'),
        palette=[[0,0,0],               # 0: background
                 [226, 124, 144],       # 1: motorway
                 [251, 192, 172],       # 2: trunk
                 [253, 215, 161],       # 3: primary
                 [246, 250, 187],       # 4: secondary
                 [255, 255, 255],       # 5: tertiary
                 [75, 238, 49],         # 6: path
                 [173, 173, 173],       # 7: under construction
                 [170, 85, 255],        # 8: train guideway
                 [120, 232, 234]]       # 9: airport runway
                 )
    
    def __init__(self, img_suffix='.png',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
