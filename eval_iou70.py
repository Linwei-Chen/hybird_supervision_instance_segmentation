#!/usr/bin/env python
"""Reproduce the paper's VOC protocol (Table 3): mAPr@0.5 / 0.7 / 0.75.

Table 3 reports COCO-style **segmentation** AP at IoU = 0.5 / 0.7 / 0.75.
A standard detectron2 `COCOEvaluator` run writes the predictions to
`<OUTPUT_DIR>/coco_instances_results.json`; the paper-era environment used a
locally patched pycocotools that printed AP@0.70 in the AP50 summary slot.
This script recomputes exactly those three numbers from a stock run:

    python eval_iou70.py <gt_json> <dt_json>

    # gt_json: the registered validation json (e.g. voc2012_val.json)
    # dt_json: coco_instances_results.json written by COCOEvaluator
"""
import sys

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def main(gt_json, dt_json):
    coco_gt = COCO(gt_json)
    dt = coco_gt.loadRes(dt_json)
    e = COCOeval(coco_gt, dt, 'segm')
    e.evaluate()
    e.accumulate()

    # precision: [T(iou) H(area) K(cls) A(maxdet) M(scores)]
    p = e.eval['precision']
    ious = e.params.iouThrs
    print("== VOC protocol (segm AP, area=all, maxdet=100) ==")
    for target in (0.5, 0.7, 0.75):
        t = int(np.argmin(np.abs(ious - target)))
        pr = p[t, :, :, 0, -1]
        pr = pr[pr > -1]
        if pr.size:
            print("segm AP@%.2f = %.1f" % (target, pr.mean() * 100))
        else:
            print("segm AP@%.2f = n/a (no predictions)" % target)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
