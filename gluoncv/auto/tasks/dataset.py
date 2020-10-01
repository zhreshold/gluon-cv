"""Dataset implementation for specific task(s)"""
import logging
from pathlib import Path
import pandas as pd
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

logger = logging.getLogger()

class ObjectDetectionDataset(pd.DataFrame):
    # preserved properties that will be copied to a new instance
    _metadata = ['dataset_type', 'classes']

    def __init__(self, data, dataset_type=None, classes=None, **kwargs):
        # dataset_type will be used to determine metrics, if None then auto resolve at runtime
        self.dataset_type = dataset_type
        # if classes is not specified(None), then infer from the annotations
        self.classes = classes
        super().__init__(data, **kwargs)

    @classmethod
    def from_iterable(cls, iterable):
        # lst is a python list with element pairs [(path, label, bbox), (path, label, bbox)...]
        raise NotImplementedError

    @classmethod
    def from_voc(cls, root, splits=None, exts=('.jpg', '.jpeg', '.png')):
        # construct from pascal VOC forma
        from ...data.pascal_voc.detection import CustomVOCDetectionBase
        rpath = Path(root).expanduser()
        img_list = []
        if splits:
            logger.debug('Use splits: %s for root: %s', str(splits), root)
            if isinstance(splits, str):
                splits = [splits]
            for split in splits:
                split_file = rpath / 'ImageSets' / 'Main' / split
                if not split_file.resolve().exists():
                    split_file = rpath / 'ImageSets' / 'Main' / (split + '.txt')
                if not split_file.resolve().exists():
                    raise FileNotFoundError(split_file)
                with split_file.open(mode='r') as fi:
                    img_list += [line.split()[0].strip() for line in fi.readlines()]
        else:
            logger.debug('No split provided, use full image list in %s', str(rpath/'JPEGImages'))
            for ext in exts:
                img_list.extend([rp.stem for rp in rpath.glob('JPEGImages/*' + ext)])
        # d = {'file': [], 'name': [],
        #      'xmin': [], 'ymin': [], 'xmax': [], 'ymax': [],
        #      'difficult': [], 'width': [], 'height': []}
        d = {'image': [], 'rois': [], 'image_attr': []}
        for stem in img_list:
            basename = stem + '.xml'
            anno_file = (rpath / 'Annotations' / basename).resolve()
            tree = ET.parse(anno_file)
            xml_root = tree.getroot()
            size = xml_root.find('size')
            im_path = xml_root.find('filename').text
            width = float(size.find('width').text)
            height = float(size.find('height').text)
            rois = []
            for obj in xml_root.iter('object'):
                try:
                    difficult = int(obj.find('difficult').text)
                except ValueError:
                    difficult = 0
                cls_name = obj.find('name').text.strip().lower()
                xml_box = obj.find('bndbox')
                xmin = max(0, float(xml_box.find('xmin').text) - 1) / width
                ymin = max(0, float(xml_box.find('ymin').text) - 1) / height
                xmax = min(width, float(xml_box.find('xmax').text) - 1) / width
                ymax = min(height, float(xml_box.find('ymax').text) - 1) / height
                if xmin >= xmax or ymin >= ymax:
                    logger.warn('Invalid bbox: {%s} for {%s}', str(xml_box), anno_file.name)
                else:
                    rois.append({'class': cls_name,
                                 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax,
                                 'difficult': difficult})
            if rois:
                d['image'].append(str(rpath / 'JPEGImages' / im_path))
                d['rois'].append(rois)
                d['image_attr'].append({'width': width, 'height': height})
        df = pd.DataFrame(d)
        return cls(df.sort_values('image').reset_index(drop=True), dataset_type='voc')

    @classmethod
    def from_coco(cls, path):
        # construct from COCO format
        raise NotImplementedError

    @classmethod
    def from_path_func(cls, fn):
        # create from a function
        raise NotImplementedError

    def pack(self):
        """Convert object-centric entries to image-centric entries.
        Where multiple entries belonging to single image can be merged to rois column.
        """
        if self.is_packed():
            return self
        rois_columns = ['class', 'xmin', 'ymin', 'xmax', 'ymax', 'difficult']
        image_attr_columns = ['width', 'height']
        new_df = self.groupby(['image'], as_index=False).agg(list).reset_index(drop=True)
        new_df['rois'] = new_df.agg(
            lambda y : [{k : y[new_df.columns.get_loc(k)][i] for k in rois_columns if k in new_df.columns} for i in range(len(y[new_df.columns.get_loc('class')]))], axis=1)
        new_df = new_df.drop(rois_columns, axis=1, errors='ignore')
        new_df['image_attr'] = new_df.agg(
            lambda y: {k : y[new_df.columns.get_loc(k)][0] for k in image_attr_columns if k in new_df.columns}, axis=1)
        new_df = new_df.drop(image_attr_columns, axis=1, errors='ignore')
        return self.__class__(new_df.reset_index(drop=True))

    def unpack(self):
        if not self.is_packed():
            return self
        new_df = self.explode('rois')
        new_df = pd.concat([new_df.drop(['rois'], axis=1), new_df['rois'].apply(pd.Series)], axis=1)
        new_df = pd.concat([new_df.drop(['image_attr'], axis=1), new_df['image_attr'].apply(pd.Series)], axis=1)
        return self.__class__(new_df.reset_index(drop=True))

    def is_packed(self):
        return 'rois' in self.columns and 'xmin' not in self.columns
