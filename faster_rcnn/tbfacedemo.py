import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, sys, cv2
import argparse
import os.path as osp
import glob

sys.path.append('/home/kfbb/TFFRCNN')
this_dir = osp.dirname(__file__)
print(this_dir)

from lib.networks.factory import get_network
from lib.fast_rcnn.config import cfg
from lib.fast_rcnn.test import im_detect
from lib.fast_rcnn.nms_wrapper import nms
from lib.utils.timer import Timer

CLASSES = ('__background__',
           'face')

log_dir=('/home/kfbb/TFFRCNN/logs/demo')


def draw_det(im, dets, thresh=0.7):
    """Draw detected bounding boxes to numpy array image."""
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(8,8))
    ax.imshow(im, aspect='equal')
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) > 0:
        for i in inds:
            bbox = dets[i, :4]
            score = dets[i, -1]

            ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                            bbox[2] - bbox[0],
                            bbox[3] - bbox[1], fill=False,
                            edgecolor='red', linewidth=3.5)
            )
            ax.text(bbox[0], bbox[1] - 2,
                    '{:.3f}'.format(score),
                    bbox=dict(facecolor='blue', alpha=0.4),
                    fontsize=14, color='white')
            print("confidence %.3f" % score)
    ax.set_title(('p({} | box) >= {:.1f}').format('face',
                                                thresh),
                fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    fig.canvas.draw()
    annot_img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    annot_img = annot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return annot_img

def build_image_summary():
    log_image_data=tf.placeholder(tf.uint8,
        [None,None,3])
    log_image_name=tf.placeholder(tf.string)
    from tensorflow.python.ops import gen_logging_ops
    from tensorflow.python.framework import ops as _ops
    log_image = gen_logging_ops._image_summary(log_image_name, 
        tf.expand_dims(log_image_data, 0),
        max_images=1)
    _ops.add_to_collection(_ops.GraphKeys.SUMMARIES, log_image)
    return log_image, log_image_data, log_image_name


def demo(sess, net, image_name, summary_writer):
    """Detect object classes in an image using pre-computed object proposals."""
    # Load the demo image
    im = cv2.imread(image_name)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # # Visualize detections for each class
    # im = im[:, :, (2, 1, 0)]
    # fig, ax = plt.subplots(figsize=(12, 12))
    # ax.imshow(im, aspect='equal')

    # image summary writer
    log_image, log_image_data, log_image_name =\
    build_image_summary()

    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        # vis_detections(im, cls, dets, ax, thresh=CONF_THRESH)
        annot_img=draw_det(im,dets,thresh=CONF_THRESH)
        im_log_name=im_name.replace('/','\\')
        log_image_summary_op = \
            sess.run( log_image,
                feed_dict={log_image_name:im_log_name,
                           log_image_data:annot_img})
        writer.add_summary(log_image_summary_op)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        default='VGGnet_test')
    parser.add_argument('--model', dest='model', help='Model path',
                        default=' ')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    if args.model == ' ' :
        print ('current path is ' + os.path.abspath(__file__))
        raise IOError(('Error: Model not found.\n'))

    # init session
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    # load network
    net = get_network(args.demo_net)
    # load model
    print ('Loading network {:s}... '.format(args.demo_net)),
    saver = tf.train.Saver()
    saver.restore(sess, args.model)
    print (' done.')

    # initialize summary writer
    writer = tf.summary.FileWriter(
        logdir=log_dir,
        graph=sess.graph,
        flush_secs=5
    )
    
    # Warmup on a dummy image
    im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _ = im_detect(sess, net, im)

    im_names = glob.glob(os.path.join(cfg.DATA_DIR, 'demo','face', '*.png')) + \
               glob.glob(os.path.join(cfg.DATA_DIR, 'demo','face','*.jpg'))

    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for {:s}'.format(im_name)
        demo(sess, net, im_name, writer)


