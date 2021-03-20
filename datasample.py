import os
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
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


# pick a random image and obtain the corresponding URL
ann_id = np.random.choice(ids)
img_id = coco.anns[ann_id]['image_id']
img = coco.loadImgs(img_id)[0]
url = img['coco_url']

# print URL and visualize corresponding image
print(url)
I = io.imread(url)
plt.axis('off')
plt.imshow(I)
plt.show()

# load and display captions
annIds = coco_caps.getAnnIds(imgIds=img['id'])
anns = coco_caps.loadAnns(annIds)
coco_caps.showAnns(anns)
