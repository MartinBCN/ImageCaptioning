import os
import sys
from pycocotools.coco import COCO
# sys.path.append('/opt/cocoapi/PythonAPI')


# initialize COCO API for instance annotations
dataDir = 'data/cocoapi'
dataType = 'val2014'
instances_annFile = os.path.join(dataDir, f'annotations/instances_{dataType}.json')
coco = COCO(instances_annFile)

# initialize COCO API for caption annotations
captions_annFile = os.path.join(dataDir, f'annotations/captions_{dataType}.json')
coco_caps = COCO(captions_annFile)

# get image ids 
ids = list(coco.anns.keys())