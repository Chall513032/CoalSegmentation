import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class CoalDataset(CustomDataset):
    """Chase_db1 dataset.

    In segmentation map annotation for Chase_db1, 0 stands for background,
    which is included in 2 categories. ``reduce_zero_label`` is fixed to False.
    The ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is fixed to
    '_1stHO.png'.
    """

    CLASSES = ('cracks', 'coal', 'gangue', 'pyrite')

    PALETTE = [[0, 0, 0], [120, 120, 120], [180, 180, 180], [255, 255, 255]]

    def __init__(self, **kwargs):
        super(MarbleDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
        assert osp.exists(self.img_dir)
