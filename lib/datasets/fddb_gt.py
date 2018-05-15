import os
import numpy as np
# import PIL
# from PIL import Image
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

data_dir='/home/kfbb/TFFRCNN/data'

def get_gt_fddb(data_dir):
    # pass
    gtdb={}
    nfold=10
    for n in xrange(nfold):
        file_name='FDDB-fold-%02d-ellipseList.txt' % (n+1)
        file_name=os.path.join(data_dir,'FDDB',
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

def vis_detections(im, class_name, gt_boxes, thresh=0.5):
    """Draw ground truth bounding ellpses."""
    inds = len(gt_boxes)
    # inds = np.where(dets[:, -1] >= thresh)[0]
    # if len(inds) == 0:
    #     return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    for i in xrange(inds):
        bbox = gt_boxes[i]
        score = 1

        ax.add_patch(
            Ellipse(
                xy=(bbox[3],bbox[4]),
                width=bbox[0]*2,
                height=bbox[1]*2,
                angle=np.degrees(bbox[2]),
                edgecolor='g',fc='None',
                linewidth=3.5
            ))


    ax.imshow(im, aspect='equal')
    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

    fig.canvas.draw()
    annot_img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    annot_img = annot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

if __name__ == '__main__':
    gtdb=get_gt_fddb(data_dir)
    print(len(gtdb))


                    