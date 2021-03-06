from __future__ import division
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import os, sys, cv2
import argparse
import os.path as osp
import glob

sys.path.append('/home/kfbb/TFFRCNN')

from lib.networks.factory import get_network
from lib.fast_rcnn.config import cfg
from lib.fast_rcnn.test import im_detect
from lib.fast_rcnn.nms_wrapper import nms
from lib.utils.timer import Timer
import scipy.io as sio

# import _init_paths
# from fast_rcnn.config import cfg
# from fast_rcnn.test import im_detect
# from fast_rcnn.nms_wrapper import nms
# import sys

NETS = {'vgg16': ('VGG16',
          'output/faster_rcnn_end2end/train/vgg16_faster_rcnn_iter_80000.caffemodel')}

def get_imdb_fddb(data_dir):
  imdb = []
  nfold = 10
  for n in xrange(nfold):
    file_name = 'FDDB-folds/FDDB-fold-%02d.txt' % (n + 1)
    file_name = os.path.join(data_dir, file_name)
    fid = open(file_name, 'r')
    image_names = []
    for im_name in fid:
      image_names.append(im_name.strip('\n'))

    imdb.append(image_names)

  return imdb

def get_gt_fddb(data_dir):
    # pass
    gtdb={}
    nfold=10
    for n in xrange(nfold):
        file_name='FDDB-fold-%02d-ellipseList.txt' % (n+1)
        file_name=os.path.join(data_dir,
            'FDDB-folds',file_name)
        with open(file_name) as f:
            gtdb_1fold=load_gt_ellipses(f)
            gtdb.update(gtdb_1fold)
    
    return gtdb

def load_gt_ellipses(f):
    gtdb={}
    lines = f.readlines()
    idx=0
    while idx < len(lines):
        image_name = lines[idx].split('\n')[0]
        # print('img name: %s' % image_name)
        image_path = os.path.join(data_dir,'FDDB','originalPics',
            image_name+'.jpg')

        idx+=1
        num_boxes = int(lines[idx])
        # print('num boxes: %d' % num_boxes)

        ellipses = np.zeros((num_boxes, 5), dtype=np.float32)

        for i in xrange(num_boxes):
            idx +=1
            coor = map(float, lines[idx].split())

            r_v   =coor[0]
            r_h   =coor[1]
            theta =coor[2]
            c_x   =coor[3]
            c_y   =coor[4]
            ellipses[i, :] = [r_v,r_h,theta,c_x,c_y]
        gtdb[image_name]=ellipses

        idx +=1 

    assert(idx == len(lines))
    return gtdb

def draw_det_gt(im, class_name, dets,gt, thresh=0.5):
    """Draw detected bounding boxes."""
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(8,8))
    ax.imshow(im, aspect='equal')
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) > 0:   
      # dpi=192
      # # get image size
      # h,w,_=im.shape
      # figsize=w/float(dpi),h/float(dpi)
      
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
                  bbox=dict(facecolor='blue', alpha=0.3),
                  fontsize=14, color='white')

    inds=len(gt)
    for i in xrange(inds):
      bbox=gt[i]
      ax.add_patch(
        Ellipse(
            xy=(bbox[3],bbox[4]),
            width=bbox[0]*2,
            height=bbox[1]*2,
            angle=np.degrees(bbox[2]),
            edgecolor='g',fc='None',
            linewidth=3.5
        ))

    ax.set_title(('p({} | box) >= {:.1f}').format(class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    # plt.draw()
    
    fig.canvas.draw()
    annot_img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    annot_img = annot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    assert(annot_img is not None)
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

def parse_args():
  """Parse input arguments."""
  parser = argparse.ArgumentParser(description='Face Detection using Faster R-CNN')
  parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
            default=0, type=int)
  parser.add_argument('--cpu', dest='cpu_mode',
            help='Use CPU mode (overrides --gpu)',
            action='store_true')
  parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
            default='VGGnet_test')
  parser.add_argument('--model', dest='model', help='Model path',
            default=' ')
  parser.add_argument('--prefix',dest='prefix', help='Log path prefix',
            default='vgg')
  args = parser.parse_args()

  return args


if __name__ == '__main__':
  cfg.TEST.HAS_RPN = True  # Use RPN for proposals
  # cfg.TEST.BBOX_REG = False

  args = parse_args()

#   prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
#               'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
#   caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
#                 NETS[args.demo_net][1])

#   prototxt = 'models/face/VGG16/faster_rcnn_end2end/test.prototxt'
#   caffemodel = NETS[args.demo_net][1]

#   if not os.path.isfile(caffemodel):
#     raise IOError(('{:s} not found.\nDid you run ./data/script/'
#              'fetch_faster_rcnn_models.sh?').format(caffemodel))

#   if args.cpu_mode:
#     caffe.set_mode_cpu()
#   else:
#     caffe.set_mode_gpu()
#     caffe.set_device(args.gpu_id)
#     cfg.GPU_ID = args.gpu_id
#   net = caffe.Net(prototxt, caffemodel, caffe.TEST)


  # global_timer=Timer()
  # global_timer.tic()
  # init session
  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
  # load network
  net = get_network(args.demo_net)
  # load model
  print ('Loading network {:s}... '.format(args.demo_net)),
  saver = tf.train.Saver()
  saver.restore(sess, args.model)
  print (' done.')
  
  data_dir = 'data/FDDB/'
  out_dir = 'output/fddb_'+args.prefix+'_res'
  log_dir = 'output/fddb_'+args.prefix+'_log'

  if not os.path.exists(out_dir):
    os.makedirs(out_dir)

  CONF_THRESH = 0.65
  NMS_THRESH = 0.15

  # summary writer
  writer = tf.summary.FileWriter(logdir=log_dir,
    graph=tf.get_default_graph(),
    flush_secs=5)

  # image summary writer
  log_image, log_image_data, log_image_name =\
    build_image_summary()
  

  imdb = get_imdb_fddb(data_dir)
  gtdb = get_gt_fddb(data_dir)

  # Warmup on a dummy image
  im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
  for i in xrange(2):
    _, _= im_detect(sess, net, im)

  nfold = len(imdb)
  for i in xrange(nfold):
    image_names = imdb[i]

    # detection file
    dets_file_name = os.path.join(out_dir, 'FDDB-det-fold-%02d.txt' % (i + 1))
    fid = open(dets_file_name, 'w')
    sys.stdout.write('%s ' % (i + 1))

    for idx, im_name in enumerate(image_names):

      gt=gtdb[im_name]
      # print('image ground truth info:')
      # print(gt)
      im = cv2.imread(os.path.join(data_dir, 'originalPics', im_name + '.jpg'))

      # Detect all object classes and regress object bounds
      timer = Timer()
      timer.tic()
      scores, boxes = im_detect(sess, net, im)
      timer.toc()
      print ('Detection took {:.3f}s for '
             'fold {:d} index {:d}').format(timer.total_time, i+1, idx)

      cls_ind = 1
      cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
      cls_scores = scores[:, cls_ind]
      dets = np.hstack((cls_boxes,
                cls_scores[:, np.newaxis])).astype(np.float32)
      keep = nms(dets, NMS_THRESH)
      dets = dets[keep, :]

      keep = np.where(dets[:, 4] > CONF_THRESH)
      dets = dets[keep]

      annot_img=draw_det_gt(im, 'face', dets, gt, CONF_THRESH)
      # im_name=str(idx)+"-"+im_name.split('/')[len(im_name.split('/'))-1]
      im_log_name=im_name.replace('/','\\')
      im_log_name=str(i+1)+'-'+str(idx)+'-'+im_log_name
      log_image_summary_op = \
        sess.run(log_image,
          feed_dict={log_image_name: im_log_name,
                     log_image_data: annot_img})
      # print('idx=%d' %idx)
      writer.add_summary(log_image_summary_op,global_step=idx)

      dets[:, 2] = dets[:, 2] - dets[:, 0] + 1
      dets[:, 3] = dets[:, 3] - dets[:, 1] + 1

      # timer.toc()
      # print ('Detection took {:.3f}s for '
      #        '{:d} object proposals').format(timer.total_time, boxes.shape[0])

      fid.write(im_name + '\n')
      fid.write(str(dets.shape[0]) + '\n')
      for j in xrange(dets.shape[0]):
        fid.write('%f %f %f %f %f\n' % (dets[j, 0], dets[j, 1], dets[j, 2], dets[j, 3], dets[j, 4]))


      if ((idx + 1) % 10) == 0:
        sys.stdout.write('%.3f of fold %d\n' % (((idx + 1) / len(image_names) * 100) ,(i+1)))
        # global_timer.toc(average=False)
        # sys.stdout.write('%ds since evaluation started\n' % global_timer.total_time)
        sys.stdout.flush()

    print ''
    fid.close()
    
    # global_timer.toc(average=False)
    # print('All evaluations took %ds' % global_timer.total_time)


  # os.system('cp ./fddb_res/*.txt ~/Code/FDDB/results')