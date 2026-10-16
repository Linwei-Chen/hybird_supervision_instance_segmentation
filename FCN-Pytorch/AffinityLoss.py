import torch
import torch.nn.functional as F
from torch import nn


def eightway_activation(x: torch.Tensor):
    """Retrieves neighboring pixels/features on the eight corners from
    a 3x3 patch.

    Args:
      x: A tensor of size [batch_size, height_in, width_in, channels]

    Returns:
      A tensor of size [batch_size, height_in, width_in, channels, 8]
    """
    # Get the number of channels in the input.
    shape_x = list(x.shape)
    if len(shape_x) != 4:
        raise ValueError('Only support for 4-D tensors!')

    # Pad at the margin.
    x = x.transpose(3, 2).transpose(2, 1)
    pad = [1, 1, 1, 1]
    x = F.pad(x, pad, mode='reflect')
    print(x.shape)
    x = x.transpose(1, 2).transpose(2, 3)
    # Get eight neighboring pixels/features.
    x_groups = [
        x[:, 1:-1, :-2, :],  # left
        x[:, 1:-1, 2:, :],  # right
        x[:, :-2, 1:-1, :],  # up
        x[:, 2:, 1:-1, :],  # down
        x[:, :-2, :-2, :],  # left-up
        x[:, 2:, :-2, :],  # left-down
        x[:, :-2, 2:, :],  # right-up
        x[:, 2:, 2:, :]  # right-down
    ]
    output = [
        c.unsqueeze(dim=-1) for c in x_groups
    ]
    output = torch.cat(output, dim=-1)
    return output


def _test_eightway_activation():
    # x = torch.randint(0, 255, [4, 3, 8, 8])
    x = torch.randn(4, 8, 8, 3)
    y = eightway_activation(x)
    print(y.shape)


def eightcorner_activation(x: torch.Tensor, size):
    """Retrieves neighboring pixels one the eight corners from a
    (2*size+1)x(2*size+1) patch.

    Args:
      x: A tensor of size [batch_size, height_in, width_in, channels]
      size: A number indicating the half size of a patch.

    Returns:
      A tensor of size [batch_size, height_in, width_in, channels, 8]
    """
    # Get the number of channels in the input.
    shape_x = list(x.shape)
    if len(shape_x) != 4:
        raise ValueError('Only support for 4-D tensors!')
    n, h, w, c = shape_x

    # Pad at the margin.
    p = size
    pad = [0, 0, p, p, p, p, 0, 0]
    x_pad = F.pad(x, pad, mode='constant', value=0)

    # Get eight corner pixels/features in the patch.
    x_groups = []
    for st_y in range(0, 2 * size + 1, size):
        for st_x in range(0, 2 * size + 1, size):
            if st_y == size and st_x == size:
                # Ignore the center pixel/feature.
                continue

            x_neighbor = x_pad[:, st_y:st_y + h, st_x:st_x + w, :]
            x_groups.append(x_neighbor)

    output = [c.unsqueeze(dim=-1) for c in x_groups]
    output = torch.cat(output, dim=-1)

    return output


def _test_eightcorner_activation():
    x = torch.randn(4, 8, 8, 3)
    y = eightcorner_activation(x, 6)
    print(y.shape)


def ignores_from_label(labels: torch.Tensor, num_classes, size):
    """Retrieves ignorable pixels from the ground-truth labels.

    This function returns a binary map in which 1 denotes ignored pixels
    and 0 means not ignored ones. For those ignored pixels, they are not
    only the pixels with label value >= num_classes, but also the
    corresponding neighboring pixels, which are on the the eight cornerls
    from a (2*size+1)x(2*size+1) patch.

    Args:
      labels: A tensor of size [batch_size, height_in, width_in], indicating
        semantic segmentation ground-truth labels.
      num_classes: A number indicating the total number of valid classes. The
        labels ranges from 0 to (num_classes-1), and any value >= num_classes
        would be ignored.
      size: A number indicating the half size of a patch.

    Return:
      A tensor of size [batch_size, height_in, width_in, 8]
    """
    # Get the number of channels in the input.
    shape_lab = list(labels.shape)
    if len(shape_lab) != 3:
        raise ValueError('Only support for 3-D label tensors!')
    n, h, w = shape_lab

    # Retrieve ignored pixels with label value >= num_classes.
    ignore = torch.gt(labels, num_classes - 1)  # NxHxW
    # print('ignore:',ignore)

    # Pad at the margin.
    p = size
    pad = [p, p, p, p, 0, 0]
    ignore_pad = F.pad(ignore, pad, mode='constant', value=1)
    # print(ignore_pad.shape)

    # Retrieve eight corner pixels from the center, where the center
    # is ignored. Note that it should be bi-directional. For example,
    # when computing AAF loss with top-left pixels, the ignored pixels
    # might be the center or the top-left ones.
    ignore_groups = []
    for st_y in range(2 * size, -1, -size):
        for st_x in range(2 * size, -1, -size):
            if st_y == size and st_x == size:
                continue
            ignore_neighbor = ignore_pad[:, st_y:st_y + h, st_x:st_x + w]
            mask = ignore_neighbor | ignore
            ignore_groups.append(mask)

    ig = 0
    for st_y in range(0, 2 * size + 1, size):
        for st_x in range(0, 2 * size + 1, size):
            # center sopt
            if st_y == size and st_x == size:
                continue
            ignore_neighbor = ignore_pad[:, st_y:st_y + h, st_x:st_x + w]
            mask = ignore_neighbor | ignore_groups[ig]
            ignore_groups[ig] = mask
            ig += 1

    ignore_groups = [
        c.unsqueeze(dim=-1) for c in ignore_groups
    ]  # NxHxWx1
    ignore = torch.cat(ignore_groups, dim=-1)  # NxHxWx8

    return ignore


def _test_ignores_from_label():
    x = torch.randint(low=0, high=25, size=(2, 4, 4))
    print(x)
    y = ignores_from_label(x, num_classes=21, size=1)
    print(y.shape)
    print(y)


def edges_from_label(labels, size, ignore_class=255):
    """Retrieves edge positions from the ground-truth labels.

    This function computes the edge map by considering if the pixel values
    are equal between the center and the neighboring pixels on the eight
    corners from a (2*size+1)*(2*size+1) patch. Ignore edges where the any
    of the paired pixels with label value >= num_classes.

    Args:
      labels: A tensor of size [batch_size, height_in, width_in, 1], indicating
        semantic segmentation ground-truth labels.
      size: A number indicating the half size of a patch.
      ignore_class: A number indicating the label value to ignore.

    Return:
      A tensor of size [batch_size, height_in, width_in, 1, 8]
    """
    # Get the number of channels in the input.
    shape_lab = labels.shape
    if len(shape_lab) != 4:
        raise ValueError('Only support for 4-D label tensors!')
    n, h, w, c = shape_lab

    # Pad at the margin.
    p = size
    pad = [0, 0, p, p, p, p, 0, 0]
    labels_pad = F.pad(labels, pad, mode='constant', value=ignore_class)

    # Get the edge by comparing label value of the center and it paired pixels.
    edge_groups = []
    for st_y in range(0, 2 * size + 1, size):
        for st_x in range(0, 2 * size + 1, size):
            if st_y == size and st_x == size:
                continue
            labels_neighbor = labels_pad[:, st_y:st_y + h, st_x:st_x + w]
            edge = torch.ne(labels_neighbor, labels)
            edge_groups.append(edge)

    edge_groups = [
        c.unsqueeze(dim=-1) for c in edge_groups
    ]  # NxHxWx1x1
    edge = torch.cat(edge_groups, dim=-1)  # NxHxWx1x8

    return edge


def _test_edges_from_label():
    x = torch.randint(low=0, high=25, size=(2, 4, 4, 1))
    print(x)
    y = edges_from_label(x, size=1)
    print(y.shape)
    print(y)


def affinity_loss(labels: torch.Tensor,
                  probs,
                  num_classes,
                  kld_margin):
    """Affinity Field (AFF) loss.

    This function computes AFF loss. There are several components in the
    function:
    1) extracts edges from the ground-truth labels.
    2) extracts ignored pixels and their paired pixels (the neighboring
       pixels on the eight corners).
    3) extracts neighboring pixels on the eight corners from a 3x3 patch.
    4) computes KL-Divergence between center pixels and their neighboring
       pixels from the eight corners.

    Args:
      labels: A tensor of size [batch_size, height_in, width_in], indicating
        semantic segmentation ground-truth labels.
      probs: A tensor of size [batch_size, height_in, width_in, num_classes],
        indicating segmentation predictions.
      num_classes: A number indicating the total number of valid classes.
      kld_margin: A number indicating the margin for KL-Divergence at edge.

    Returns:
      Two 1-D tensors value indicating the loss at edge and non-edge.
    """
    # Compute ignore map (e.g, label of 255 and their paired pixels).
    labels = labels.squeeze(dim=-1)  # NxHxW
    ignore = ignores_from_label(labels, num_classes, 1)  # NxHxWx8
    not_ignore = ~ignore
    # print(f'gnore:{ignore}')
    # print(f'not ignore:{not_ignore}')
    not_ignore = not_ignore.unsqueeze(dim=3)  # NxHxWx1x8, 1 indicate not ignore

    # Compute edge map.
    labels_no_ignore = torch.where(labels < num_classes, labels, torch.zeros(labels.shape, device=labels.device).long())
    one_hot_lab = F.one_hot(labels_no_ignore, num_classes=num_classes)  # NxHxWxC-one-hot
    edge = edges_from_label(one_hot_lab, 1, 255)  # NxHxWxCx8

    # Remove ignored pixels from the edge/non-edge.
    edge = edge & not_ignore

    not_edge = (~ edge) & not_ignore

    edge_indices = (edge.reshape(-1) == 1)
    # print(f'edge:{edge}')
    # print(f'edge_indices:{edge_indices.sum()}{edge_indices}')
    not_edge_indices = (not_edge.reshape(-1) == 1)

    # Extract eight corner from the center in a patch as paired pixels.
    probs_paired = eightcorner_activation(probs, 1)  # NxHxWxCx8
    probs = probs.unsqueeze(dim=-1)  # NxHxWxCx1

    bot_epsilon = 1e-4  # tf.constant(1e-4, name='bot_epsilon')
    top_epsilon = 1.0  # tf.constant(1.0, name='top_epsilon')

    neg_probs = torch.clamp(
        1 - probs, bot_epsilon, top_epsilon)
    probs = torch.clamp(
        probs, bot_epsilon, top_epsilon)
    neg_probs_paired = torch.clamp(
        1 - probs_paired, bot_epsilon, top_epsilon)
    probs_paired = torch.clamp(
        probs_paired, bot_epsilon, top_epsilon)

    # Compute KL-Divergence.
    kldiv = probs_paired * torch.log(probs_paired / probs)
    kldiv += neg_probs_paired * torch.log(neg_probs_paired / neg_probs)
    not_edge_loss = kldiv
    edge_loss = torch.max(torch.zeros(size=kldiv.shape, device=kldiv.device), kld_margin - kldiv)

    not_edge_loss = not_edge_loss.reshape(-1)
    # not_edge_loss = torch.gather(not_edge_loss, dim=0, index=not_edge_indices)
    not_edge_loss = not_edge_loss[not_edge_indices]
    edge_loss = edge_loss.reshape(-1)
    # edge_loss = torch.gather(edge_loss, dim=0, index=edge_indices)
    edge_loss = edge_loss[edge_indices]

    return edge_loss, not_edge_loss


def _test_affinity_loss():
    labels = torch.randint(low=0, high=9, size=(2, 4, 4, 1))
    probs = torch.randn(size=(2, 4, 4, 9))
    num_classes = 9
    kld_margin = 5
    edge_loss, not_edge_loss = affinity_loss(labels, probs, num_classes, kld_margin)
    print(edge_loss, not_edge_loss)


def adaptive_affinity_loss(labels,
                           one_hot_lab,
                           probs,
                           size,
                           num_classes,
                           kld_margin,
                           w_edge,
                           w_not_edge):
    """Adaptive affinity field (AAF) loss.

    This function computes AAF loss. There are three components in the function:
    1) extracts edges from the ground-truth labels.
    2) extracts ignored pixels and their paired pixels (usually the eight corner
       pixels).
    3) extracts eight corner pixels/predictions from the center in a
       (2*size+1)x(2*size+1) patch
    4) computes KL-Divergence between center pixels and their paired pixels (the
       eight corner).
    5) imposes adaptive weightings on the loss.

    Args:
      labels: A tensor of size [batch_size, height_in, width_in, 1], indicating
        semantic segmentation ground-truth labels.
      one_hot_lab: A tensor of size [batch_size, height_in, width_in, num_classes]
        which is the ground-truth labels in the form of one-hot vector.
      probs: A tensor of size [batch_size, height_in, width_in, num_classes],
        indicating segmentation predictions.
      size: A number indicating the half size of a patch.
      num_classes: A number indicating the total number of valid classes. The
      kld_margin: A number indicating the margin for KL-Divergence at edge.
      w_edge: A number indicating the weighting for KL-Divergence at edge.
      w_not_edge: A number indicating the weighting for KL-Divergence at non-edge.

    Returns:
      Two 1-D tensors value indicating the loss at edge and non-edge.
    """
    # Compute ignore map (e.g, label of 255 and their paired pixels).
    labels = labels.squeeze(-1)  # NxHxW
    ignore = ignores_from_label(labels, num_classes, size)  # NxHxWx8
    not_ignore = ~ignore
    not_ignore = not_ignore.unsqueeze(dim=3)  # NxHxWx1x8

    # Compute edge map.
    edge = edges_from_label(one_hot_lab, size, 255)  # NxHxWxCx8

    # Remove ignored pixels from the edge/non-edge.
    edge = edge & not_ignore
    not_edge = (~edge) & not_ignore

    edge_indices = (edge.reshape(-1) == 1)
    not_edge_indices = (not_edge.reshape(-1) == 1)

    # Extract eight corner from the center in a patch as paired pixels.
    probs_paired = eightcorner_activation(probs, size)  # NxHxWxCx8
    probs = probs.unsqueeze(dim=-1)  # NxHxWxCx1
    bot_epsilon = 1e-4  # tf.constant(1e-4, name='bot_epsilon')
    top_epsilon = 1.0  # tf.constant(1.0, name='top_epsilon')

    neg_probs = torch.clamp(
        1 - probs, bot_epsilon, top_epsilon)
    neg_probs_paired = torch.clamp(
        1 - probs_paired, bot_epsilon, top_epsilon)
    probs = torch.clamp(
        probs, bot_epsilon, top_epsilon)
    probs_paired = torch.clamp(
        probs_paired, bot_epsilon, top_epsilon)

    # Compute KL-Divergence.
    kldiv = probs_paired * torch.log(probs_paired / probs)
    kldiv += neg_probs_paired * torch.log(neg_probs_paired / neg_probs)
    edge_loss = torch.max(torch.zeros(size=kldiv.shape, device=kldiv.device), kld_margin - kldiv)
    not_edge_loss = kldiv  # NxHxWxCx1

    # Impose weights on edge/non-edge losses.
    one_hot_lab = one_hot_lab.unsqueeze(dim=-1)
    # print(w_edge.shape, one_hot_lab.shape)
    w_edge = torch.sum(w_edge * one_hot_lab, dim=3, keepdim=True)  # NxHxWx1x1
    w_not_edge = torch.sum(w_not_edge * one_hot_lab, dim=3, keepdim=True)  # NxHxWx1x1

    edge_loss *= w_edge
    not_edge_loss *= w_not_edge

    not_edge_loss = not_edge_loss.reshape(-1)
    not_edge_loss = not_edge_loss[not_edge_indices]
    edge_loss = edge_loss.reshape(-1)
    edge_loss = edge_loss[edge_indices]

    return edge_loss, not_edge_loss


class AdaptiveAffinityLoss(nn.Module):
    def __init__(self, args, kld_margin, size):
        super().__init__()
        self.w_not_edge = nn.Parameter(torch.ones(1, 1, 1, args.n_class, 1), requires_grad=True)
        self.w_edge = nn.Parameter(torch.ones(1, 1, 1, args.n_class, 1), requires_grad=True)
        self.kld_margin = kld_margin
        self.num_classes = args.n_class
        self.size = size

    def forward(self, labels, one_hot_lab, probs):
        """Adaptive affinity field (AAF) loss.

        This function computes AAF loss. There are three components in the function:
        1) extracts edges from the ground-truth labels.
        2) extracts ignored pixels and their paired pixels (usually the eight corner
           pixels).
        3) extracts eight corner pixels/predictions from the center in a
           (2*size+1)x(2*size+1) patch
        4) computes KL-Divergence between center pixels and their paired pixels (the
           eight corner).
        5) imposes adaptive weightings on the loss.

        Args:
          labels: A tensor of size [batch_size, height_in, width_in, 1], indicating
            semantic segmentation ground-truth labels.
          one_hot_lab: A tensor of size [batch_size, height_in, width_in, num_classes]
            which is the ground-truth labels in the form of one-hot vector.
          probs: A tensor of size [batch_size, height_in, width_in, num_classes],
            indicating segmentation predictions.
          size: A number indicating the half size of a patch.
          num_classes: A number indicating the total number of valid classes. The
          kld_margin: A number indicating the margin for KL-Divergence at edge.
          w_edge: A number indicating the weighting for KL-Divergence at edge.
          w_not_edge: A number indicating the weighting for KL-Divergence at non-edge.

        Returns:
          Two 1-D tensors value indicating the loss at edge and non-edge.
        """

        # Compute ignore map (e.g, label of 255 and their paired pixels).

        labels = labels.squeeze(-1)  # NxHxW
        ignore = ignores_from_label(labels, self.num_classes, self.size)  # NxHxWx8
        not_ignore = ~ignore
        not_ignore = not_ignore.unsqueeze(dim=3)  # NxHxWx1x8

        # Compute edge map.
        edge = edges_from_label(one_hot_lab, self.size, 255)  # NxHxWxCx8

        # Remove ignored pixels from the edge/non-edge.
        edge = edge & not_ignore
        not_edge = (~edge) & not_ignore

        edge_indices = (edge.reshape(-1) == 1)
        not_edge_indices = (not_edge.reshape(-1) == 1)

        # Extract eight corner from the center in a patch as paired pixels.
        probs_paired = eightcorner_activation(probs, self.size)  # NxHxWxCx8
        probs = probs.unsqueeze(dim=-1)  # NxHxWxCx1
        bot_epsilon = 1e-4  # tf.constant(1e-4, name='bot_epsilon')
        top_epsilon = 1.0  # tf.constant(1.0, name='top_epsilon')

        neg_probs = torch.clamp(
            1 - probs, bot_epsilon, top_epsilon)
        neg_probs_paired = torch.clamp(
            1 - probs_paired, bot_epsilon, top_epsilon)
        probs = torch.clamp(
            probs, bot_epsilon, top_epsilon)
        probs_paired = torch.clamp(
            probs_paired, bot_epsilon, top_epsilon)

        # Compute KL-Divergence.
        kldiv = probs_paired * torch.log(probs_paired / probs)
        kldiv += neg_probs_paired * torch.log(neg_probs_paired / neg_probs)
        edge_loss = torch.max(torch.zeros(size=kldiv.shape, device=kldiv.device), self.kld_margin - kldiv)
        not_edge_loss = kldiv  # NxHxWxCx1

        # Impose weights on edge/non-edge losses.
        one_hot_lab = one_hot_lab.unsqueeze(dim=-1)
        # print(w_edge.shape, one_hot_lab.shape)
        w_edge = torch.sum(self.w_edge * one_hot_lab, dim=3, keepdim=True)  # NxHxWx1x1
        w_not_edge = torch.sum(self.w_not_edge * one_hot_lab, dim=3, keepdim=True)  # NxHxWx1x1

        edge_loss *= w_edge
        not_edge_loss *= w_not_edge

        not_edge_loss = not_edge_loss.reshape(-1)
        not_edge_loss = not_edge_loss[not_edge_indices]
        edge_loss = edge_loss.reshape(-1)
        edge_loss = edge_loss[edge_indices]

        return edge_loss, not_edge_loss


def _test_adaptive_affinity_loss():
    labels = torch.randint(low=0, high=3, size=(2, 2, 2, 1))
    probs = torch.randn(size=(2, 2, 2, 3))
    num_classes = 3
    kld_margin = 5
    edge_loss, not_edge_loss = affinity_loss(labels, probs, num_classes, kld_margin)
    print(edge_loss, not_edge_loss)


if __name__ == '__main__':
    _test_affinity_loss()
    pass
