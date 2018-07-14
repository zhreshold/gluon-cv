"""Train YOLO"""
import argparse
import os
import logging
import time
import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet import autograd
import gluoncv as gcv
from gluoncv import data as gdata
from gluoncv import utils as gutils
from gluoncv.model_zoo import get_model
from gluoncv.data.batchify import Tuple, Stack, Pad
from gluoncv.data.transforms.presets.yolo import YOLO3DefaultTrainTransform
from gluoncv.data.transforms.presets.yolo import YOLO3DefaultValTransform
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric
from gluoncv.utils.metrics.coco_detection import COCODetectionMetric
from gluoncv.utils import LRScheduler
from gluoncv.model_zoo.yolo.yolo_target import YOLOV3TargetMerger

def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLO networks.')
    parser.add_argument('--network', type=str, default='darknet53',
                        help="Base network name which serves as feature extraction base.")
    parser.add_argument('--data-shape', type=int, default=416,
                        help="Input data shape, use 416, 608...")
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Training mini-batch size')
    parser.add_argument('--dataset', type=str, default='voc',
                        help='Training dataset. Now support voc.')
    parser.add_argument('--num-workers', '-j', dest='num_workers', type=int,
                        default=4, help='Number of data workers, you can use larger '
                        'number to accelerate data loading, if you CPU and GPUs are powerful.')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Training epochs.')
    parser.add_argument('--resume', type=str, default='',
                        help='Resume from previously saved parameters if not None. '
                        'For example, you can resume from ./ssd_xxx_0123.params')
    parser.add_argument('--start-epoch', type=int, default=0,
                        help='Starting epoch for resuming, default is 0 for new training.'
                        'You can specify it to 100 for example to start from 100 epoch.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate, default is 0.001')
    parser.add_argument('--lr-decay', type=float, default=0.1,
                        help='decay rate of learning rate. default is 0.1.')
    parser.add_argument('--lr-decay-epoch', type=str, default='160,180',
                        help='epoches at which learning rate decays. default is 160,180.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum, default is 0.9')
    parser.add_argument('--wd', type=float, default=0.0005,
                        help='Weight decay, default is 5e-4')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='Logging mini-batch interval. Default is 100.')
    parser.add_argument('--save-prefix', type=str, default='',
                        help='Saving parameter prefix')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='Saving parameters epoch interval, best model will always be saved.')
    parser.add_argument('--val-interval', type=int, default=1,
                        help='Epoch interval for validation, increase the number will reduce the '
                             'training time if validation is slow.')
    parser.add_argument('--seed', type=int, default=233,
                        help='Random seed to be fixed.')
    parser.add_argument('--num-samples', type=int, default=-1,
                        help='Training images. Use -1 to automatically get the number.')
    args = parser.parse_args()
    return args

# from mxnet.gluon.loss import _apply_weighting, _reshape_like
# class SigmoidBinaryCrossEntropyLoss(gluon.loss.Loss):
#     r"""The cross-entropy loss for binary classification. (alias: SigmoidBCELoss)
#
#     BCE loss is useful when training logistic regression. If `from_sigmoid`
#     is False (default), this loss computes:
#
#     .. math::
#
#         prob = \frac{1}{1 + \exp(-{pred})}
#
#         L = - \sum_i {label}_i * \log({prob}_i) +
#             (1 - {label}_i) * \log(1 - {prob}_i)
#
#     If `from_sigmoid` is True, this loss computes:
#
#     .. math::
#
#         L = - \sum_i {label}_i * \log({pred}_i) +
#             (1 - {label}_i) * \log(1 - {pred}_i)
#
#
#     `pred` and `label` can have arbitrary shape as long as they have the same
#     number of elements.
#
#     Parameters
#     ----------
#     from_sigmoid : bool, default is `False`
#         Whether the input is from the output of sigmoid. Set this to false will make
#         the loss calculate sigmoid and BCE together, which is more numerically
#         stable through log-sum-exp trick.
#     weight : float or None
#         Global scalar weight for loss.
#     batch_axis : int, default 0
#         The axis that represents mini-batch.
#
#
#     Inputs:
#         - **pred**: prediction tensor with arbitrary shape
#         - **label**: target tensor with values in range `[0, 1]`. Must have the
#           same size as `pred`.
#         - **sample_weight**: element-wise weighting tensor. Must be broadcastable
#           to the same shape as pred. For example, if pred has shape (64, 10)
#           and you want to weigh each sample in the batch separately,
#           sample_weight should have shape (64, 1).
#
#     Outputs:
#         - **loss**: loss tensor with shape (batch_size,). Dimenions other than
#           batch_axis are averaged out.
#     """
#     def __init__(self, from_sigmoid=False, weight=None, batch_axis=0, **kwargs):
#         super(SigmoidBinaryCrossEntropyLoss, self).__init__(weight, batch_axis, **kwargs)
#         self._from_sigmoid = from_sigmoid
#
#     def hybrid_forward(self, F, pred, label, sample_weight=None):
#         label = _reshape_like(F, label, pred)
#         if not self._from_sigmoid:
#             # We use the stable formula: max(x, 0) - x * z + log(1 + exp(-abs(x)))
#             loss = F.relu(pred) - pred * label + F.Activation(-F.abs(pred), act_type='softrelu')
#         else:
#             loss = -(F.log(pred+1e-12)*label + F.log(1.-pred+1e-12)*(1.-label))
#         loss = _apply_weighting(F, loss, self._weight, sample_weight)
#         assert (loss.asnumpy() >= 0).all(), "negative loss! {}".format(sample_weight.asnumpy()[np.where(loss.asnumpy() < 0)])
#         return F.mean(loss, axis=self._batch_axis, exclude=True)

def get_dataset(dataset, args):
    if dataset.lower() == 'voc':
        train_dataset = gdata.VOCDetection(
            splits=[(2007, 'trainval'), (2012, 'trainval')])
        val_dataset = gdata.VOCDetection(
            splits=[(2007, 'test')])
        val_metric = VOC07MApMetric(iou_thresh=0.5, class_names=val_dataset.classes)
    elif dataset.lower() == 'coco':
        train_dataset = gdata.COCODetection(splits='instances_train2017')
        val_dataset = gdata.COCODetection(splits='instances_val2017', skip_empty=False)
        val_metric = COCODetectionMetric(
            val_dataset, args.save_prefix + '_eval', cleanup=True,
            data_shape=(args.data_shape, args.data_shape))
    else:
        raise NotImplementedError('Dataset: {} not implemented.'.format(dataset))
    if args.num_samples < 0:
        args.num_samples = len(train_dataset)
    return train_dataset, val_dataset, val_metric

def get_dataloader(net, train_dataset, val_dataset, data_shape, batch_size, num_workers):
    """Get dataloader."""
    width, height = data_shape, data_shape
    batchify_fn = Tuple(*([Stack() for _ in range(6)] + [Pad(axis=0, pad_val=-1) for _ in range(2)]))  # stack image, all targets generated
    train_loader = gluon.data.DataLoader(
        train_dataset.transform(YOLO3DefaultTrainTransform(width, height, net)),
        batch_size, True, batchify_fn=batchify_fn, last_batch='rollover', num_workers=num_workers)
    val_batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
    val_loader = gluon.data.DataLoader(
        val_dataset.transform(YOLO3DefaultValTransform(width, height)),
        batch_size, False, batchify_fn=val_batchify_fn, last_batch='keep', num_workers=num_workers)
    return train_loader, val_loader

def save_params(net, best_map, current_map, epoch, save_interval, prefix):
    current_map = float(current_map)
    if current_map > best_map[0]:
        best_map[0] = current_map
        net.save_parameters('{:s}_best.params'.format(prefix, epoch, current_map))
        with open(prefix+'_best_map.log', 'a') as f:
            f.write('\n{:04d}:\t{:.4f}'.format(epoch, current_map))
    if save_interval and epoch % save_interval == 0:
        net.save_parameters('{:s}_{:04d}_{:.4f}.params'.format(prefix, epoch, current_map))

def validate(net, val_data, ctx, eval_metric):
    """Test on validation dataset."""
    eval_metric.reset()
    # set nms threshold and topk constraint
    net.set_nms(nms_thresh=0.45, nms_topk=400)
    net.hybridize()
    for batch in val_data:
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
        det_bboxes = []
        det_ids = []
        det_scores = []
        gt_bboxes = []
        gt_ids = []
        gt_difficults = []
        for x, y in zip(data, label):
            # get prediction results
            ids, scores, bboxes = net(x)
            det_ids.append(ids)
            det_scores.append(scores)
            # clip to image size
            det_bboxes.append(bboxes.clip(0, batch[0].shape[2]))
            # split ground truths
            gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
            gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
            gt_difficults.append(y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None)

        # update metric
        eval_metric.update(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficults)
    return eval_metric.get()


def debug_viz(objness_t, center_t, scale_t, weight_t, class_t, featmaps, anchors, offsets, imgs, gt_boxes, gt_ids, net):
    import matplotlib.pyplot as plt
    secs = [0, 169 * 3, 845 * 3, 3549 * 3]

    for b in range(imgs.shape[0]):
        img = imgs[b].transpose((1, 2, 0)) * mx.nd.array([0.229, 0.224, 0.225]) + mx.nd.array([0.485, 0.456, 0.406])
        img = (img * 255).asnumpy()
        bboxes = []
        scores = []
        labels = []
        for i in range(3):
            x = featmaps[i]
            a = anchors[i][0][0].asnumpy()
            o = offsets[i]
            begin = secs[i]
            end = secs[i+1]
            obj_tt = objness_t[b].asnumpy()[begin:end, 0]
            center_tt = center_t[b].asnumpy()[begin:end, :]
            scale_tt = scale_t[b].asnumpy()[begin:end, :]
            weight_tt = weight_t[b].asnumpy()[begin:end, :]
            class_tt = class_t[b].asnumpy()[begin:end, :]
            for j in range(class_tt.shape[0]):
                if (class_tt[j, :] <= 0).all():
                    continue
                loc_y = (j // 3) // x.shape[3]
                loc_x = (j // 3) % x.shape[3]
                bx = (loc_x + center_tt[j, 0]) / x.shape[3] * img.shape[1]
                by = (loc_y + center_tt[j, 1]) / x.shape[2] * img.shape[0]
                bw = np.exp(scale_tt[j, 0]) * a[j % 3, 0]
                bh = np.exp(scale_tt[j, 1]) * a[j % 3, 1]
                print('otx', center_tt[j, 0], 'oty', center_tt[j, 1], 'otw', scale_tt[j, 0], 'oth', scale_tt[j, 1])
                bboxes.append([bx - bw/2, by - bh/2, bx + bw/ 2, by + bh / 2])
                scores.append(1.)
                labels.append(np.where(class_tt[j, :] > 0)[0][0])
        gtb = gt_boxes[b].asnumpy()
        gtb = gtb[np.where(gtb > -1)].reshape(-1, 4)
        gti = gt_ids[b].asnumpy()
        gti = gti[np.where(gti > -1)].reshape(-1, 1)
        print(gtb, bboxes)
        gcv.utils.viz.plot_bbox(img, np.array(bboxes), np.array(scores), np.array(labels), class_names=net.classes)
        # gcv.utils.viz.plot_bbox(img, gtb, labels=gti, class_names=net.classes)
        plt.show()
    # raise

def train(net, train_data, val_data, eval_metric, args):
    """Training pipeline"""
    net.collect_params().reset_ctx(ctx)
    lr_scheduler = LRScheduler(mode='step',
                               baselr=args.lr,
                               niters=args.num_samples // args.batch_size,
                               nepochs=args.epochs,
                               step=[int(i) for i in args.lr_decay_epoch.split(',')],
                               step_factor=float(args.lr_decay),
                               warmup_epochs=max(1, 1000 // (args.num_samples // args.batch_size)),
                               warmup_mode='linear')

    trainer = gluon.Trainer(
        net.collect_params(), 'sgd',
        {'wd': args.wd, 'momentum': args.momentum, 'lr_scheduler': lr_scheduler})

    # targets
    target_merger = YOLOV3TargetMerger()
    sigmoid_ce = gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=False)
    # sce = SigmoidBinaryCrossEntropyLoss(from_sigmoid=False)
    l1_loss = gluon.loss.L1Loss()

    # metrics
    obj_metrics = mx.metric.Loss('ObjLoss')
    center_metrics = mx.metric.Loss('BoxCenterLoss')
    scale_metrics = mx.metric.Loss('BoxScaleLoss')
    cls_metrics = mx.metric.Loss('ClassLoss')

    # set up logger
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_file_path = args.save_prefix + '_train.log'
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fh = logging.FileHandler(log_file_path)
    logger.addHandler(fh)
    logger.info(args)
    logger.info('Start training from [Epoch {}]'.format(args.start_epoch))
    best_map = [0]
    for epoch in range(args.start_epoch, args.epochs):
        tic = time.time()
        btic = time.time()
        net.hybridize()
        for i, batch in enumerate(train_data):
            batch_size = batch[0].shape[0]
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            fixed_targets = [gluon.utils.split_and_load(batch[it], ctx_list=ctx, batch_axis=0) for it in range(1, 8)]
            sum_losses = []
            obj_losses = []
            center_losses = []
            scale_losses = []
            cls_losses = []
            with autograd.record():
                for ix, x in enumerate(data):
                    tmp = net(x)
                    box_preds, anchors, offsets, featmaps, box_centers, box_scales, objness, cls_preds = net(x)
                    gt_boxes = fixed_targets[-2][ix]
                    gt_ids = fixed_targets[-1][ix]
                    dynamic_targets = net.target_generator(x, featmaps, anchors, offsets, box_preds, gt_boxes, gt_ids)
                    objness_t, center_t, scale_t, weight_t, class_t, class_mask = target_merger(*(list(dynamic_targets) + [ft[ix] for ft in fixed_targets[:-2]]))
                    # debug_viz(objness_t, center_t, scale_t, weight_t, class_t, featmaps, anchors, offsets, x, gt_boxes, gt_ids, net)
                    obj_loss = sigmoid_ce(objness, objness_t, objness_t >= 0) * objness.size / batch_size
                    center_loss = sigmoid_ce(box_centers, center_t, weight_t) * box_centers.size / batch_size
                    scale_loss = l1_loss(box_scales, scale_t, weight_t) * box_scales.size / batch_size
                    cls_loss = sigmoid_ce(cls_preds, class_t, class_mask) * cls_preds.size / batch_size
                    sum_losses.append(obj_loss + center_loss + scale_loss + cls_loss)
                    obj_losses.append(obj_loss)
                    center_losses.append(center_loss)
                    scale_losses.append(scale_loss)
                    cls_losses.append(cls_loss)
                autograd.backward(sum_losses)
            lr_scheduler.update(i, epoch)
            trainer.step(batch_size)
            obj_metrics.update(0, obj_losses)
            center_metrics.update(0, center_losses)
            scale_metrics.update(0, scale_losses)
            cls_metrics.update(0, cls_losses)
            if args.log_interval and not (i + 1) % args.log_interval:
                name1, loss1 = obj_metrics.get()
                name2, loss2 = center_metrics.get()
                name3, loss3 = scale_metrics.get()
                name4, loss4 = cls_metrics.get()
                logger.info('[Epoch {}][Batch {}], LR: {:.2E}, Speed: {:.3f} samples/sec, {}={:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}'.format(
                    epoch, i, trainer.learning_rate, batch_size/(time.time()-btic), name1, loss1, name2, loss2, name3, loss3, name4, loss4))
            btic = time.time()

        name1, loss1 = obj_metrics.get()
        name2, loss2 = center_metrics.get()
        name3, loss3 = scale_metrics.get()
        name4, loss4 = cls_metrics.get()
        logger.info('[Epoch {}] Training cost: {:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}'.format(
            epoch, (time.time()-tic), name1, loss1, name2, loss2, name3, loss3, name4, loss4))
        if not (epoch + 1) % args.val_interval:
            # consider reduce the frequency of validation to save time
            map_name, mean_ap = validate(net, val_data, ctx, eval_metric)
            val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
            logger.info('[Epoch {}] Validation: \n{}'.format(epoch, val_msg))
            current_map = float(mean_ap[-1])
        else:
            current_map = 0.
        save_params(net, best_map, current_map, epoch, args.save_interval, args.save_prefix)

if __name__ == '__main__':
    args = parse_args()
    # fix seed for mxnet, numpy and python builtin random generator.
    gutils.random.seed(args.seed)

    # training contexts
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = ctx if ctx else [mx.cpu()]

    # network
    net_name = '_'.join(('yolo3', str(args.data_shape), args.network, args.dataset))
    args.save_prefix += net_name
    net = get_model(net_name, pretrained_base=True)
    if args.resume.strip():
        net.load_params(args.resume.strip())
    else:
        for param in net.collect_params().values():
            if param._data is not None:
                continue
            param.initialize()

    # training data
    train_dataset, val_dataset, eval_metric = get_dataset(args.dataset, args)
    train_data, val_data = get_dataloader(
        net, train_dataset, val_dataset, args.data_shape, args.batch_size, args.num_workers)

    # training
    train(net, train_data, val_data, eval_metric, args)
