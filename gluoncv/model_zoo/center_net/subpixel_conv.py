"""Subpixel convolution with learnable weights."""
import warnings
import mxnet as mx
from mxnet import gluon
from mxnet.base import numeric_types
from mxnet import cpu
from mxnet.gluon import nn
from mxnet.gluon.nn.conv_layers import _Conv
from mxnet.util import is_np_array
from ..model_zoo import get_model
from ...nn.block import DUC


class SubPixelConv2D(_Conv):
    def __init__(self, channels, kernel_size, strides=(1, 1), padding=(0, 0),
                 dilation=(1, 1), groups=1, layout='NCHW',
                 activation=None, use_bias=True, weight_initializer=None,
                 bias_initializer='zeros', in_channels=0, **kwargs):
        assert layout in ('NCHW', 'NHWC'), "Only supports 'NCHW' and 'NHWC' layout for now"
        if isinstance(kernel_size, numeric_types):
            kernel_size = (kernel_size,)*2
        assert len(kernel_size) == 2, "kernel_size must be a number or a list of 2 ints"
        op_name = kwargs.pop('op_name', 'Convolution')
        if is_np_array():
            op_name = 'convolution'
        super(SubPixelConv2D, self).__init__(
            channels, kernel_size, strides, padding, dilation, groups, layout,
            in_channels, activation, use_bias, weight_initializer, bias_initializer,
            op_name, **kwargs)
        self.mask = self.params.get('mask', shape=(1, 1, kernel_size[0], kernel_size[1]),
                                    init='ones',
                                    allow_deferred_init=False)

    def hybrid_forward(self, F, x, weight, mask, bias=None):
        if is_np_array():
            F = F.npx
            FF = F.np
        masks = [mask, F.flip(mask, axis=3), F.flip(mask, axis=2)]
        masks.append(F.flip(masks[-1], axis=3))
        acts = []
        for m in masks:
            w = F.broadcast_mul(weight, m)
            if bias is None:
                act = getattr(F, self._op_name)(x, w, name='fwd', **self._kwargs)
            else:
                act = getattr(F, self._op_name)(x, w, bias, name='fwd', **self._kwargs)
            acts.append(act)
        # merge into H and W
        if is_np_array():
            raise NotImplementedError()
        else:
            act1 = F.concat(acts[0], acts[2], dim=-1).reshape((0, 0, 0, 2, -1)).reshape((0, 0, -3, -1))
            act2 = F.concat(acts[1], acts[3], dim=-1).reshape((0, 0, 0, 2, -1)).reshape((0, 0, -3, -1))
            act = F.concat(act1.expand_dims(-1), act2.expand_dims(-1), dim=-1)
            act = act.reshape((0, 0, 0, -1))
        if self.act is not None:
            act = self.act(act)
        return act


class SPConvResnet(nn.HybridBlock):
    """spconvolutional ResNet.

    Parameters
    ----------
    base_network : str
        Name of the base feature extraction network.
    spconv_filters : list of int
        Number of filters for spconv layers.
    pretrained_base : bool
        Whether load pretrained base network.
    norm_layer : mxnet.gluon.nn.HybridBlock
        Type of Norm layers, can be BatchNorm, SyncBatchNorm, GroupNorm, etc.
    norm_kwargs : dict
        Additional kwargs for `norm_layer`.

    """
    def __init__(self, base_network='resnet18_v1b',
                 spconv_filters=(256, 128, 64),
                 pretrained_base=True, norm_layer=nn.BatchNorm, norm_kwargs=None,
                 **kwargs):
        super(SPConvResnet, self).__init__(**kwargs)
        assert 'resnet' in base_network
        net = get_model(base_network, pretrained=pretrained_base)
        self._norm_layer = norm_layer
        self._norm_kwargs = norm_kwargs if norm_kwargs is not None else {}
        if 'v1b' in base_network:
            feat = nn.HybridSequential()
            feat.add(*[net.conv1,
                       net.bn1,
                       net.relu,
                       net.maxpool,
                       net.layer1,
                       net.layer2,
                       net.layer3,
                       net.layer4])
            self.base_network = feat
        else:
            raise NotImplementedError('Only v1 variants of resnet are supported so far.')
        with self.name_scope():
            self.spconv = self._make_spconv_layer(spconv_filters)

    def _make_spconv_layer(self, num_filters):
        # pylint: disable=unused-variable
        """Make spconv layers using the configs"""

        layers = nn.HybridSequential('spconv_')
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.base_network.initialize()
        in_planes = self.base_network(mx.nd.zeros((1, 3, 256, 256))).shape[1]
        for planes in num_filters:
            layers.add(nn.Conv2D(channels=planes,
                                 kernel_size=3,
                                 strides=1,
                                 padding=1,
                                 in_channels=in_planes))
            layers.add(self._norm_layer(momentum=0.9, **self._norm_kwargs))
            layers.add(nn.Activation('relu'))
            layers.add(SubPixelConv2D(channels=planes,
                                      kernel_size=3,
                                      strides=1,
                                      padding=1,
                                      in_channels=planes))
            layers.add(self._norm_layer(momentum=0.9, **self._norm_kwargs))
            layers.add(nn.Activation('relu'))
            in_planes = planes
        return layers

    def hybrid_forward(self, F, x):
        # pylint: disable=arguments-differ
        """HybridForward"""
        y = self.base_network(x)
        out = self.spconv(y)
        return out


class DUCResnet(nn.HybridBlock):
    """DUC ResNet.

    Parameters
    ----------
    base_network : str
        Name of the base feature extraction network.
    duc_filters : list of int
        Number of filters for duc layers.
    pretrained_base : bool
        Whether load pretrained base network.
    norm_layer : mxnet.gluon.nn.HybridBlock
        Type of Norm layers, can be BatchNorm, SyncBatchNorm, GroupNorm, etc.
    norm_kwargs : dict
        Additional kwargs for `norm_layer`.

    """
    def __init__(self, base_network='resnet18_v1b',
                 duc_filters=(256*4, 128*4, 64*4),
                 pretrained_base=True, norm_layer=nn.BatchNorm, norm_kwargs=None,
                 **kwargs):
        super(DUCResnet, self).__init__(**kwargs)
        assert 'resnet' in base_network
        net = get_model(base_network, pretrained=pretrained_base)
        self._norm_layer = norm_layer
        self._norm_kwargs = norm_kwargs if norm_kwargs is not None else {}
        if 'v1b' in base_network:
            feat = nn.HybridSequential()
            feat.add(*[net.conv1,
                       net.bn1,
                       net.relu,
                       net.maxpool,
                       net.layer1,
                       net.layer2,
                       net.layer3,
                       net.layer4])
            self.base_network = feat
        else:
            raise NotImplementedError('Only v1 variants of resnet are supported so far.')
        with self.name_scope():
            self.duc = self._make_duc_layer(duc_filters)

    def _make_duc_layer(self, num_filters):
        # pylint: disable=unused-variable
        """Make spconv layers using the configs"""

        layers = nn.HybridSequential('spconv_')
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.base_network.initialize()
        in_planes = self.base_network(mx.nd.zeros((1, 3, 256, 256))).shape[1]
        for planes in num_filters:
            layers.add(nn.Conv2D(channels=planes,
                                 kernel_size=3,
                                 strides=1,
                                 padding=1,
                                 in_channels=in_planes))
            layers.add(self._norm_layer(momentum=0.9, **self._norm_kwargs))
            layers.add(nn.Activation('relu'))
            layers.add(DUC(planes, upscale_factor=2))
            in_planes = planes // 4
        return layers

    def hybrid_forward(self, F, x):
        # pylint: disable=arguments-differ
        """HybridForward"""
        y = self.base_network(x)
        out = self.duc(y)
        return out

def get_spconv_resnet(base_network, pretrained=False, ctx=cpu(), **kwargs):
    """Get resnet with spconv layers.

    Parameters
    ----------
    base_network : str
        Name of the base feature extraction network.
    pretrained : bool
        Whether load pretrained base network.
    ctx : mxnet.Context
        mx.cpu() or mx.gpu()
    pretrained : type
        Description of parameter `pretrained`.
    Returns
    -------
    nn.HybridBlock
        Network instance of resnet with spconv layers

    """
    net = SPConvResnet(base_network=base_network, pretrained_base=pretrained,
                       **kwargs)
    with warnings.catch_warnings(record=True) as _:
        warnings.simplefilter("always")
        net.initialize()
    net.collect_params().reset_ctx(ctx)
    return net

def resnet18_v1b_spconv(**kwargs):
    """Resnet18 v1b model with spconv layers.

    Returns
    -------
    HybridBlock
        A Resnet18 v1b model with spconv layers.

    """
    return get_spconv_resnet('resnet18_v1b', **kwargs)

def resnet50_v1b_spconv(**kwargs):
    """Resnet18 v1b model with spconv layers.

    Returns
    -------
    HybridBlock
        A Resnet50 v1b model with spconv layers.

    """
    return get_spconv_resnet('resnet50_v1b', **kwargs)

def resnet101_v1b_spconv(**kwargs):
    """Resnet18 v1b model with spconv layers.

    Returns
    -------
    HybridBlock
        A Resnet101 v1b model with spconv layers.

    """
    return get_spconv_resnet('resnet101_v1b', **kwargs)

def get_duc_resnet(base_network, pretrained=False, ctx=cpu(), **kwargs):
    """Get resnet with spconv layers.

    Parameters
    ----------
    base_network : str
        Name of the base feature extraction network.
    pretrained : bool
        Whether load pretrained base network.
    ctx : mxnet.Context
        mx.cpu() or mx.gpu()
    pretrained : type
        Description of parameter `pretrained`.
    Returns
    -------
    nn.HybridBlock
        Network instance of resnet with spconv layers

    """
    net = DUCResnet(base_network=base_network, pretrained_base=pretrained,
                    **kwargs)
    with warnings.catch_warnings(record=True) as _:
        warnings.simplefilter("always")
        net.initialize()
    net.collect_params().reset_ctx(ctx)
    return net

def resnet18_v1b_duc(**kwargs):
    """Resnet18 v1b model with spconv layers.

    Returns
    -------
    HybridBlock
        A Resnet18 v1b model with spconv layers.

    """
    return get_duc_resnet('resnet18_v1b', **kwargs)

def resnet50_v1b_duc(**kwargs):
    """Resnet18 v1b model with spconv layers.

    Returns
    -------
    HybridBlock
        A Resnet50 v1b model with spconv layers.

    """
    return get_duc_resnet('resnet50_v1b', **kwargs)

def resnet101_v1b_duc(**kwargs):
    """Resnet18 v1b model with spconv layers.

    Returns
    -------
    HybridBlock
        A Resnet101 v1b model with spconv layers.

    """
    return get_duc_resnet('resnet101_v1b', **kwargs)
