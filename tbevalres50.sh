#!/bin/bash
python faster_rcnn/tbface_detect_on_fddb.py --prefix res50n --model output/wider/wider/res50_frcnn_wider_iter_70000.ckpt --net Resnet50_test
