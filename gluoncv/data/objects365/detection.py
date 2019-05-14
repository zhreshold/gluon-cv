"""Objects 365 detection dataset."""
from __future__ import absolute_import
from __future__ import division
import os
from ..mscoco.detection import COCODetection


class Objects365Detection(COCODetection):
    """Objects 365 detection dataset.

    Parameters
    ----------
    root : str, default '~/.mxnet/datasets/objects365'
        Path to folder storing the dataset.
    splits : list of str, default ['val']
        Json annotations name.
        Candidates can be: train, val.
    transform : callable, default None
        A function that takes data and label and transforms them. Refer to
        :doc:`./transforms` for examples.

        A transform function for object detection should take label into consideration,
        because any geometric modification will require label to be modified.
    min_object_area : float
        Minimum accepted ground-truth area, if an object's area is smaller than this value,
        it will be ignored.
    skip_empty : bool, default is True
        Whether skip images with no valid object. This should be `True` in training, otherwise
        it will cause undefined behavior.
    use_crowd : bool, default is True
        Whether use boxes labeled as crowd instance.

    """
    CLASSES = ['human', 'sneakers', 'chair', 'hat', 'lamp', 'bottle', 'cabinet/shelf', 'cup',
               'car', 'glasses', 'picture/frame', 'desk', 'handbag', 'street lights', 'book',
               'plate', 'helmet', 'leather shoes', 'pillow', 'glove', 'potted plant', 'bracelet',
               'flower', 'monitor', 'storage box', 'plants pot/vase', 'bench', 'wine glass',
               'boots', 'dining table', 'umbrella', 'boat', 'flag', 'speaker', 'trash bin/can',
               'stool', 'backpack', 'sofa', 'belt', 'carpet', 'basket', 'towel/napkin', 'slippers',
               'bowl', 'barrel/bucket', 'coffee table', 'suv', 'toy', 'tie', 'bed',
               'traffic light', 'pen/pencil', 'microphone', 'sandals', 'canned', 'necklace',
               'mirror', 'faucet', 'bicycle', 'bread', 'high heels', 'ring', 'van', 'watch',
               'combine with bowl', 'sink', 'horse', 'fish', 'apple', 'traffic sign', 'camera',
               'candle', 'stuffed animal', 'cake', 'motorbike/motorcycle', 'wild bird', 'laptop',
               'knife', 'cellphone', 'paddle', 'truck', 'cow', 'power outlet', 'clock', 'drum',
               'fork', 'bus', 'hanger', 'nightstand', 'pot/pan', 'sheep', 'guitar', 'traffic cone',
               'tea pot', 'keyboard', 'tripod', 'hockey stick', 'fan', 'dog', 'spoon',
               'blackboard/whiteboard', 'balloon', 'air conditioner', 'cymbal', 'mouse',
               'telephone', 'pickup truck', 'orange', 'banana', 'airplane', 'luggage', 'skis',
               'soccer', 'trolley', 'oven', 'remote', 'combine with glove', 'paper towel',
               'refrigerator', 'train', 'tomato', 'machinery vehicle', 'tent', 'shampoo/shower gel',
               'head phone', 'lantern', 'donut', 'cleaning products', 'sailboat', 'tangerine',
               'pizza', 'kite', 'computer box', 'elephant', 'toiletries', 'gas stove', 'broccoli',
               'toilet', 'stroller', 'shovel', 'baseball bat', 'microwave', 'skateboard',
               'surfboard', 'surveillance camera', 'gun', 'Life saver', 'cat', 'lemon',
               'liquid soap', 'zebra', 'duck', 'sports car', 'giraffe', 'pumpkin',
               'Accordion/keyboard/piano', 'radiator', 'converter', 'tissue ', 'carrot',
               'washing machine', 'vent', 'cookies', 'cutting/chopping board', 'tennis racket',
               'candy', 'skating and skiing shoes', 'scissors', 'folder', 'baseball', 'strawberry',
               'bow tie', 'pigeon', 'pepper', 'coffee machine', 'bathtub', 'snowboard', 'suitcase',
               'grapes', 'ladder', 'pear', 'american football', 'basketball', 'potato',
               'paint brush', 'printer', 'billiards', 'fire hydrant', 'goose', 'projector',
               'sausage', 'fire extinguisher', 'extension cord', 'facial mask', 'tennis ball',
               'chopsticks', 'Electronic stove and gas stove', 'pie', 'frisbee', 'kettle',
               'hamburger', 'golf club', 'cucumber', 'clutch', 'blender', 'tong', 'slide',
               'hot dog', 'toothbrush', 'facial cleanser', 'mango', 'deer', 'egg', 'violin',
               'marker', 'ship', 'chicken', 'onion', 'ice cream', 'tape', 'wheelchair', 'plum',
               'bar soap', 'scale', 'watermelon', 'cabbage', 'router/modem', 'golf ball',
               'pine apple', 'crane', 'fire truck', 'peach', 'cello', 'notepaper', 'tricycle',
               'toaster', 'helicopter', 'green beans', 'brush', 'carriage', 'cigar', 'earphone',
               'penguin', 'hurdle', 'swing', 'radio', 'CD', 'parking meter', 'swan', 'garlic',
               'french fries', 'horn', 'avocado', 'saxophone', 'trumpet', 'sandwich', 'cue',
               'kiwi fruit', 'bear', 'fishing rod', 'cherry', 'tablet', 'green vegetables', 'nuts',
               'corn', 'key', 'screwdriver', 'globe', 'broom', 'pliers', 'hammer', 'volleyball',
               'eggplant', 'trophy', 'board eraser', 'dates', 'rice', 'tape measure/ruler',
               'dumbbell', 'hamimelon', 'stapler', 'camel', 'lettuce', 'goldfish', 'meat balls',
               'medal', 'toothpaste', 'antelope', 'shrimp', 'rickshaw', 'trombone', 'pomegranate',
               'coconut', 'jellyfish', 'mushroom', 'calculator', 'treadmill', 'butterfly',
               'egg tart', 'cheese', 'pomelo', 'pig', 'race car', 'rice cooker', 'tuba',
               'crosswalk sign', 'papaya', 'hair dryer', 'green onion', 'chips', 'dolphin',
               'sushi', 'urinal', 'donkey', 'electric drill', 'spring rolls', 'tortoise/turtle',
               'parrot', 'flute', 'measuring cup', 'shark', 'steak', 'poker card', 'binoculars',
               'llama', 'radish', 'noodles', 'mop', 'yak', 'crab', 'microscope', 'barbell',
               'Bread/bun', 'baozi', 'lion', 'red cabbage', 'polar bear', 'lighter', 'mangosteen',
               'seal', 'comb', 'eraser', 'pitaya', 'scallop', 'pencil case', 'saw',
               'table tennis  paddle', 'okra', 'starfish', 'monkey', 'eagle', 'durian', 'rabbit',
               'game board', 'french horn', 'ambulance', 'asparagus', 'hoverboard', 'pasta',
               'target', 'hotair balloon', 'chainsaw', 'lobster', 'iron', 'flashlight']

    def __init__(self, root=os.path.join('~', '.mxnet', 'datasets', 'objects365'),
                 splits=('val',), transform=None, min_object_area=0,
                 skip_empty=True, use_crowd=True):
        super(Objects365Detection, self).__init__(root=root, splits=splits, transform=transform,
                                                  min_object_area=min_object_area,
                                                  skip_empty=skip_empty, use_crowd=use_crowd)

    @property
    def annotation_dir(self):
        """
        The subdir for annotations. Default is '.'(root folder of objects365 dataset)
        For example, a coco format json file will be searched as
        'root/xxx.json'
        You can override if custom dataset don't follow the same pattern
        """
        return '.'
