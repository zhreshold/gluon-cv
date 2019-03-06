"""Contrib, cached detection dataset."""
import os
import shelve
import mxnet as mx
from .detection import COCODetection
from ..transforms.presets.rcnn import FasterRCNNDefaultTrainTransform


class CachedFasterRCNNCOCODetection(COCODetection):
    def __init__(self, cache_file, sizes=((600, 1000), (800, 1300)), **kwargs):
        assert kwargs.get('transform', None) is None
        super(CachedFasterRCNNCOCODetection, self).__init__(**kwargs)
        self._cache_file = os.path.abspath(os.path.expanduser(cache_file))
        self._shelve = shelve.open(self._cache_file)
        self._sizes = sizes
        assert isinstance(self._sizes, (list, tuple))
        for sz in self._sizes:
            assert len(sz) == 2
        trans_fn = [FasterRCNNDefaultTrainTransform(
            short=min_s, max_size=max_s, flip_p=0, **kwargs) for min_s, max_s in sizes]
        trans_fn_flip = [FasterRCNNDefaultTrainTransform(
            short=min_s, max_size=max_s, flip_p=1, **kwargs) for min_s, max_s in sizes]
        self._trans_fns = trans_fn + trans_fn_flip
        self._num_trans = len(self._trans_fns)

    def __del__(self):
        self._shelve.close()

    def transform(self, fn, lazy=True):
        raise ValueError("Cached Detection dataset don't support transform")

    def __len__(self):
        return super(CachedFasterRCNNCOCODetection, self).__len__() * self._num_trans

    def __getitem__(self, idx):
        real_idx = idx // self._num_trans
        img_path = self._items[real_idx]
        img = mx.image.imread(img_path, 1)
        trans_fn = self._trans_fns[idx % self._num_trans]
        key = '{}_{}_{}_{}'.format(img_path, trans_fn._short, trans_fn._max_size, trans_fn._flip_p)
        if key in self._shelve:
            # grab cached label, read image
            label, meta = self._shelve[key]
            height = meta['height']
            width = meta['width']
            flipped = meta['flipped']

            img = mx.image.imresize(img, width, height, interp=1)
            if flipped:
                img = mx.nd.flip(image, axis=1)
            # to tensor
            img = mx.nd.image.to_tensor(img)
            img = mx.nd.image.normalize(img, mean=trans_fn._mean, std=trans_fn._std)
        else:
            # generate image and labels
            label = self._labels[idx]
            rets = trans_fn(img, label)
            img = rets[0]
            label = rets[1:]
            # save to cache
            _, _, height, width = img.shape
            flipped = trans_fn._flip_p > 0
            meta = {'height':height, 'width':width, 'flipped':flipped}
            self._shelve[key] = (label, meta)
            # self._shelve.sync()

        return [img] + label
