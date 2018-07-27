from pycocotools.coco import COCO
import cv2
from torchvision import transforms
import torch
from nltk import word_tokenize

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
dtype = torch.float

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

def get_images(dataDir, dataType):
    """
    Returns information for all MS COCO images from the dataType (e.g. 'train2014') dataset.
    """
    annFile = '%s/annotations/instances_%s.json' % (dataDir, dataType)
    coco = COCO(annFile)
    imgIds = coco.getImgIds()
    return coco.loadImgs(imgIds)

def get_image_data(dataDir, dataType, images):
    """
    Returns tensor filled with the data from images, which are first resized and cropped.
    """
    process = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    imgData = torch.zeros(len(images), 3, 224, 224, device=device, dtype=dtype)

    for i in range(len(images)):
        imgData[i] = process(cv2.imread('%s/datasets/mscoco/%s/%s' % (dataDir, dataType, images[i]['file_name']))) # resize image and convert to tensor

    return imgData

def get_captions(dataDir, dataType, images):
    """
    Returns list of tokenized captions for each image in images.
    """
    capFile = '%s/annotations/captions_%s.json' % (dataDir, dataType)
    caps = COCO(capFile)

    captions = []

    for i in range(len(images)):
        annIds = caps.getAnnIds(imgIds=images[i]['id'])  # retrieve captions for image
        anns = caps.loadAnns(annIds)
        imageCaps = []
        for ann in anns:
            imageCaps.append(tokenize(ann['caption']))
        captions.append(imageCaps)

    return captions

def create_encoding(captions):
    """
    Maps each word present in captions to an integer. Returns the vocabulary list and this mapping.
    """
    wordCounts = {}
    for imgCaps in captions:
        for cap in imgCaps:
            for word in cap:
                if word in wordCounts:
                    wordCounts[word] += 1
                else:
                    wordCounts[word] = 1

    vocab = list(wordCounts.keys())
    vocabEncoding = {}  # map each word to an integer
    for i in range(len(vocab)):
        vocabEncoding[vocab[i]] = i

    return (vocab, vocabEncoding)