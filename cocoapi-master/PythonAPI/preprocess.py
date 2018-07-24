from pycocotools.coco import COCO
import io
import requests
from PIL import Image
from torchvision import transforms
import torch
from nltk import word_tokenize
import pickle

def is_number(s):
    """Returns True if the string s is any kind of number (includes decimals)."""
    try:
        float(s)
        return True
    except ValueError:
        return False

def tokenize(caption):
    """
    Returns a list of words (tokens) in the given caption.
    """
    caption = caption.lower()
    tokens = ['<start>'] + word_tokenize(caption) + ['<end>']
    for i in range(len(tokens)):
        if is_number(tokens[i]):
            tokens[i] = '<num>' # replaces all numbers with number token
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

# store captions
save_captions = open('captions.pickle', 'wb')
pickle.dump(captions, save_captions)
save_captions.close()

