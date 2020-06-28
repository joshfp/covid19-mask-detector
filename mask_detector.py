import sys

from facenet_pytorch import MTCNN
from facenet_pytorch.models.utils import detect_face

import torch
import numpy as np
import mmcv, cv2
from PIL import Image, ImageDraw
from IPython import display
import imageio

from fastai2.vision.all import *
from torchvision import transforms

sys.path.append('../../..')

OK_COLOR = (0, 255, 0)
HIGHLIGHT_COLOR = (255, 0, 0)
BB_ALPHA = 50
BB_WIDTH = 2

class ImageClassifier():
    def __init__(self, learner_fn, image_sz):
        self.tfms = transforms.Compose([transforms.Resize(image_sz), 
                                   transforms.CenterCrop(image_sz), 
                                   transforms.ToTensor()])
        
        learn = load_learner(learner_fn)
        self.model = learn.model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.classes = learn.dls.vocab
        
    def predict(self, images):
        if not images:
            return []
        x = torch.stack([self.tfms(img) for img in images]).to(self.device)
        preds = self.model(x).argmax(dim=1).cpu()
        return [self.classes[pred] for pred in preds]

class MaskDetector():
    def __init__(self, learner_fn, image_sz=128):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('Running on device: {}'.format(self.device))
        
        self.mask_clf = ImageClassifier(learner_fn, image_sz)
        
    def _face_detector(self, image):
        mtcnn = MTCNN(keep_all=True, device=self.device, min_face_size=40, thresholds=[0.6, 0.7, 0.7])
        # Detect faces
        boxes, _ = mtcnn.detect(image)

        return boxes
        
    def _get_faces(self, img, boxes, extra_size=0.):
        img_w, img_h = img.size
        patches = []
        for l,u,r,b in boxes:
            extra_w, extra_h = (r - l) * extra_size, (b - u) * extra_size
            l, r = max(l - extra_w/2, 0), min(r + extra_w/2, img_w)
            u, b = max(u - extra_h/2, 0), min(b + extra_h/2, img_h)
            patch = img.crop((l, u, r, b))
            patches.append(patch)
        return patches
    
    def _get_masks(self, faces):
        return [pred=='mask' for pred in self.mask_clf.predict(faces)]
    
    def _plot_bboxes(self, boxes, masks, draw):
        for box, mask in zip(boxes, masks):
            color = OK_COLOR if mask else HIGHLIGHT_COLOR
            draw.rectangle(box, outline=color, width=BB_WIDTH, fill=color+(BB_ALPHA,))
        return draw
        
    def detect(self, image):
        image_draw = image.copy()
        draw = ImageDraw.Draw(image_draw, 'RGBA')
        
        # 1: Get boxes
        boxes = self._face_detector(image)

        # 2: Get faces
        faces = self._get_faces(image, boxes, extra_size=0.6)
        
        # 3: Get masks
        masks = self._get_masks(faces)
        
        # 4: Plot boxes
        draw = self._plot_bboxes(boxes, masks, draw)
        
        return image_draw