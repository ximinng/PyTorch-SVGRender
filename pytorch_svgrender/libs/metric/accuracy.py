# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Author: XiMing Xing
# Description:


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k.

    Args
        output: logits or probs (num of batch, num of classes)
        target: (num of batch, 1) or (num of batch, )
        topk: list of returned k

    refer: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    maxK = max(topk)  # get k in top-k
    batch_size = target.size(0)

    _, pred = output.topk(k=maxK, dim=1, largest=True, sorted=True)  # pred: [num of batch, k]
    pred = pred.t()  # pred: [k, num of batch]

    # [1, num of batch] -> [k, num_of_batch] : bool
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res  # np.shape(res): [k, 1]
