#!/usr/bin/env python3
# coding: utf-8

__author__ = 'cleardusk'

"""
The pipeline of 3DDFA prediction: given one image, predict the 3d face vertices, 68 landmarks and visualization.

[todo]
1. CPU optimization: https://pmchojnacki.wordpress.com/2018/10/07/slow-pytorch-cpu-performance
"""

import torch
import torchvision.transforms as transforms
import mobilenet_v1
import numpy as np
import cv2
import dlib
from utils.ddfa import ToTensorGjz, NormalizeGjz, str2bool
import scipy.io as sio
from utils.inference import get_suffix, parse_roi_box_from_landmark, crop_img, predict_68pts, dump_to_ply, dump_vertex, \
    draw_landmarks, predict_dense, parse_roi_box_from_bbox, get_colors, write_obj_with_colors
from utils.cv_plot import plot_pose_box
from utils.estimate_pose import parse_pose
from utils.render import get_depths_image, cget_depths_image, cpncc
from utils.paf import gen_img_paf
import argparse
import torch.backends.cudnn as cudnn

STD_SIZE = 120
draw_shiftbits = 4
draw_multiplier = 1 << 4


def main(args):
    # 1. load pre-tained model
    checkpoint_fp = 'models/phase1_wpdc_vdc.pth.tar'
    arch = 'mobilenet_1'

    checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
    model = getattr(mobilenet_v1, arch)(num_classes=62)  # 62 = 12(pose) + 40(shape) +10(expression)

    model_dict = model.state_dict()
    # because the model is trained by multiple gpus, prefix module should be removed
    for k in checkpoint.keys():
        model_dict[k.replace('module.', '')] = checkpoint[k]
    model.load_state_dict(model_dict)
    if args.mode == 'gpu':
        cudnn.benchmark = True
        model = model.cuda()
    model.eval()

    face_detector = dlib.get_frontal_face_detector()

    # 3. forward
    tri = sio.loadmat('visualize/tri.mat')['tri']
    transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])

    last_frame_lmks = []

    vc = cv2.VideoCapture(0)
    success, frame = vc.read()
    while success:
        roi_box = []
        
        if len(last_frame_lmks) == 0:
            rects = face_detector(frame, 1)
            for rect in rects:
                bbox = [rect.left(), rect.top(), rect.right(), rect.bottom()]
                roi_box.append(parse_roi_box_from_bbox(bbox))
        else:
            for lmk in last_frame_lmks:
                roi_box.append(parse_roi_box_from_landmark(lmk))
        
        this_frame_lmk = []
        for box in roi_box:
            img_to_net = crop_img(frame, box)
            img_to_net = cv2.resize(img_to_net, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)
            input = transform(img_to_net).unsqueeze(0)
            with torch.no_grad():
                if args.mode == 'gpu':
                    input = input.cuda()
                param = model(input)
                param = param.squeeze().cpu().numpy().flatten().astype(np.float32)
            this_frame_lmk.append(predict_68pts(param, box))

        last_frame_lmks = this_frame_lmk
        
        for lmk in last_frame_lmks:
            for p in lmk.T:
                cv2.circle(frame, (int(round(p[0] * draw_multiplier)), int(round(p[1] * draw_multiplier))), draw_multiplier, (255,0,0), 1, cv2.LINE_AA, draw_shiftbits)
        cv2.imshow("3ddfa video demo", frame)
        cv2.waitKey(1)
        success, frame = vc.read()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3DDFA inference pipeline')
    parser.add_argument('-m', '--mode', default='cpu', type=str, help='gpu or cpu mode')

    args = parser.parse_args()
    main(args)
