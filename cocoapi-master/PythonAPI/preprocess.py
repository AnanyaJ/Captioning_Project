from pycocotools.coco import COCO
import io
import requests
from PIL import Image
from torchvision import transforms
import torch
from nltk import word_tokenize

def tokenize(caption):
    """
    Returns a list of words (tokens) in the given caption.
    """
    caption = caption.lower()
    tokens = word_tokenize(caption)
    return tokens

dataDir = '..'
dataType = 'train2014'
annFile = '%s/annotations/instances_%s.json' % (dataDir, dataType)
capFile = '%s/annotations/captions_%s.json' % (dataDir, dataType)
caps = COCO(capFile)

# retrieve images
coco = COCO(annFile)
imgIds = coco.getImgIds()
images = coco.loadImgs(imgIds)

print('%d Images Loaded.' % len(images))

process = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

img_data = torch.zeros(len(images), 3, 224, 224, dtype=torch.float)
captions = []

for i in range(len(images)):
    img_url = images[i]['flickr_url']
    img_data[i] = process(Image.open(io.BytesIO(requests.get(img_url).content))) # resize image and convert to tensor

    annIds = caps.getAnnIds(imgIds=images[i]['id']) # retrieve captions for image
    anns = caps.loadAnns(annIds)
    imageCaps = []
    for ann in anns:
        imageCaps.append(tokenize(ann['caption']))
    captions.append(imageCaps)

# store image data
torch.save(img_data, 'img_data.pt')

