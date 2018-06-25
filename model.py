from helper import *

# FEATURE LOSS NETWORK
def lossnet(input, n_layers=14, training=True, reuse=False, norm_type="SBN",
               ksz=3, base_channels=32, blk_channels=5):
    layers = []

    if norm_type == "NM": # ADAPTIVE BATCH NORM
        norm_fn = nm
    elif norm_type == "SBN": # BATCH NORM
        norm_fn = slim.batch_norm
    else: # NO LAYER NORMALIZATION
        norm_fn = None

    for id in range(n_layers):

        n_channels = base_channels * (2 ** (id // blk_channels)) # UPDATE CHANNEL COUNT

        if id == 0:
            net = slim.conv2d(input, n_channels, [1, ksz], activation_fn=lrelu, normalizer_fn=norm_fn, stride=[1, 2],
                              scope='loss_conv_%d' % id, padding='SAME', reuse=reuse)
            layers.append(net)
        elif id < n_layers - 1:
            net = slim.conv2d(layers[-1], n_channels, [1, ksz], activation_fn=lrelu, normalizer_fn=norm_fn,
                              stride=[1, 2], scope='loss_conv_%d' % id, padding='SAME', reuse=reuse)
            layers.append(net)
        else:
            net = slim.conv2d(layers[-1], n_channels, [1, ksz], activation_fn=lrelu, normalizer_fn=norm_fn,
                              scope='loss_conv_%d' % id, padding='SAME', reuse=reuse)
            layers.append(net)

    return layers


def featureloss(target, current, loss_weights, loss_layers, n_layers=14, norm_type="SBN", base_channels=32, blk_channels=5):

    feat_current = lossnet(current, reuse=False, n_layers=n_layers, norm_type=norm_type,
                         base_channels=base_channels, blk_channels=blk_channels)

    feat_target = lossnet(target, reuse=True, n_layers=n_layers, norm_type=norm_type,
                         base_channels=base_channels, blk_channels=blk_channels)

    loss_vec = [0]
    for id in range(loss_layers):
        loss_vec.append(l1_loss(feat_current[id], feat_target[id]) / loss_weights[id])

    for id in range(1,loss_layers+1):
        loss_vec[0] += loss_vec[id]

    return loss_vec

# ENHANCEMENT NETWORK
def senet(input, n_layers=13, training=True, reuse=False, norm_type="NM",
          ksz=3, n_channels=32):

    if norm_type == "NM": # ADAPTIVE BATCH NORM
        norm_fn = nm
    elif norm_type == "SBN": # BATCH NORM
        norm_fn = slim.batch_norm
    else: # NO LAYER NORMALIZATION
        norm_fn = None

    for id in range(n_layers):

        if id == 0:
            net = slim.conv2d(input, n_channels, [1, ksz], activation_fn=lrelu,
                              normalizer_fn=norm_fn, scope='se_conv_%d' % id,
                              padding='SAME', reuse=reuse)
        else:
            net, pad_elements = signal_to_dilated(net, n_channels=n_channels, dilation=2 ** id)
            net = slim.conv2d(net, n_channels, [1, ksz], activation_fn=lrelu,
                              normalizer_fn=norm_fn, scope='se_conv_%d' % id,
                              padding='SAME', reuse=reuse)
            net = dilated_to_signal(net, n_channels=n_channels, pad_elements=pad_elements)

    net = slim.conv2d(net, n_channels, [1, ksz], activation_fn=lrelu,
                      normalizer_fn=norm_fn, scope='se_conv_last',
                      padding='SAME', reuse=reuse)

    output = slim.conv2d(net, 1, [1, 1], activation_fn=None,
                         scope='se_fc_last', padding='SAME', reuse=reuse)

    return output