#!/bin/bash
# python faster_rcnn/tbface_detect_on_fddb.py --model output/wider_vgg/wider/vgg_frcnn_wider_iter_70000.ckpt --prefix vgg
python faster_rcnn/tbface_detect_on_fddb.py --model output/wider_nvgg/wider/nvgg_frcnn_wider_iter_70000.ckpt --prefix nnvgg
