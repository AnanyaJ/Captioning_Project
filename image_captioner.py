import torch
from torchvision import models
import random
from nltk.translate.bleu_score import sentence_bleu
import sys
sys.path.insert(0, 'cocoapi-master/PythonAPI/')
from preprocess import get_images, get_image_data, get_captions, create_encoding

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
dtype = torch.float

def shuffle(images, captions):
    """
    Randomly shuffles items in captions and reorders images according to new caption order.

    Returns new list of images and captions.
    """

    for i in range(len(captions)): # keep track of which image index each caption corresponds to
        captions[i] = (captions[i], i)

    random.shuffle(captions)
    imagesOrdered = []

    for i in range(len(captions)):
        imagesOrdered.append(images[captions[i][1]]) # re-order images to match new caption order
        captions[i] = captions[i][0]

    return (imagesOrdered, captions)

def sort_descending_length(images, captions):
    """
    Sorts captions by decreasing length (number of words) and reorders images according to new caption order.

    Returns new tensor with image data along with list of sorted captions.
    """

    capsOrdered = []
    for i in range(len(captions)): # keep track of which image index each caption corresponds to
        capsOrdered.append(captions[i] + [i])

    capsOrdered.sort(reverse=True, key=len) # sort captions by length (descending)
    imagesOrdered = torch.zeros(images.size(), device=device, dtype=dtype)

    for i in range(len(capsOrdered)):
        imagesOrdered[i] = images[capsOrdered[i][-1]] # re-order images to match new caption order
        capsOrdered[i] = capsOrdered[i][:-1]

    return (imagesOrdered, capsOrdered)

def encode(captions, vocabSize, vocabEncoding):
    """
    Converts captions into tensors based on vocabEncoding.

    Arguments:
        captions - list of lists of words, sorted by decreasing length
        vocabSize - number of words in vocabulary
        vocabEncoding - mapping from words in vocabulary to integers
    Returns:
        onehot - tensor with captions, whose words are represented by one-hot vectors (excludes end tokens),
                padded to shape (num_captions, max_caption_length, vocabSize)
        encoded - tensor with caption words represented by integers (excludes start tokens), padded to shape (num_captions, max_caption_length)
        lengths - lengths (descending) of captions
    """

    onehot = torch.zeros(len(captions), len(captions[0]), vocabSize, device=device, dtype=dtype)
    encoded = torch.zeros(len(captions), len(captions[0]), device=device, dtype=dtype)
    lengths = []

    for i in range(len(captions)):
        caption = captions[i]
        lengths.append(len(caption)-1)
        for w in range(lengths[i]):
            if caption[w] in vocabEncoding:
                onehot[i][w][vocabEncoding[caption[w]]] = 1
                encoded[i][w] = vocabEncoding[caption[w]]
            else: # unknown word (appears less than 5 times in training data)
                onehot[i][w][vocabEncoding['<unk>']] = 1
                encoded[i][w] = vocabEncoding['<unk>']

    return (onehot[:, :-1, :], encoded[:, 1:].long(), lengths)

def prepare_data(images, captions, vocabSize, vocabEncoding):
    """
    Prepares input (x1, x2, lengths) and output (y) data from a batch of images (tensor) and captions (list).
    """

    for i in range(len(captions)):
        captions[i] = random.choice(captions[i])  # randomly choose one caption for each image

    (x1, captions) = sort_descending_length(images, captions)

    (x2, y, lengths) = encode(captions, vocabSize, vocabEncoding)

    return (x1, x2, lengths, y)

def evaluate(images, captions, model, dataType, vocab, numTests=50, numCapsPerTest=5):
    """
    Uses model to generate captions for images and evaluates these generated captions by computing a BLEU score.

    Arguments:
        images - list of images to get data from
        captions - list with a set of tokenized captions for each image
        model - model used to generate captions from images
        dataType - dataset to get image data from
        vocab - list of words in vocabulary (word index = integer representing word in encoding)
        numTests - number of images to generate captions for
        numCapsPerTest - number of captions to generate per image
    """

    model.eval()

    startTokenVector = torch.zeros(1, 1, len(vocab), device=device, dtype=dtype)
    startTokenVector[vocabEncoding['<start>']] = 1

    imageData = get_image_data('..', dataType, images[:numTests])

    bleuTotal = 0

    for i in range(numTests):
        maxBLEU = 0 # keep track of only the maximum BLEU score across all generated captions for a particular image

        for c in range(numCapsPerTest):
            cap = []
            y_pred = model(imageData[i].unsqueeze(0))
            for wordIndex in y_pred:
                if vocab[wordIndex] not in {'<start>', '<end>', '<unk>', '<num>'}:
                    cap.append(vocab[wordIndex]) # decode words with vocab

            bleuScore = sentence_bleu(captions[i], cap)
            maxBLEU = max(maxBLEU, bleuScore)

        bleuTotal += maxBLEU

    print('BLEU-4 Score (%d Images): %f' % (numTests, bleuTotal/numTests))

    return bleuTotal/numTests

class Captioner(torch.nn.Module):

    def __init__(self, vocabSize, endTokenIndex, maxCapLength=20):
        """
        Model for image caption generation with a CNN encoder to understand the image followed by an LSTM decoder to produce a corresponding caption.

        Arguments:
            vocabSize - the number of words in the vocabulary
            endTokenIndex - the integer that the end token was mapped to
            maxCapLength - maximum desired length of generated captions
        """
        super(Captioner, self).__init__()

        self.endToken = endTokenIndex
        self.vocabSize = vocabSize
        self.maxCapLength = maxCapLength

        vgg = models.vgg16_bn(pretrained=True)
        newClassifier = torch.nn.Sequential(*list(vgg.classifier.children())[:-1]) # remove last FC layer
        vgg.classifier = newClassifier

        for param in vgg.parameters():  # freeze CNN parameters (except for last layer)
            param.requires_grad = False

        self.vgg = vgg
        self.linear1 = torch.nn.Linear(4096, 512) # new last layer of CNN
        self.linear2 = torch.nn.Linear(self.vocabSize, 512) # produces word embeddings
        self.lstm = torch.nn.LSTM(512, 512, batch_first=True)
        self.linear3 = torch.nn.Linear(512, self.vocabSize)
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, x1, x2=torch.Tensor(), lengths=[]):
        """
        Implements the forward pass of the caption generator.

        Arguments:
            x1 - tensor representing image
            x2 - tensor representing sequence of words in caption, of shape (batch_size, max_seq_length, num_classes)
        Returns:
            y_pred - (if training) tensor with sequence of probability distributions representing word predictions
                     (otherwise) list of integers corresponding to predicted words
        """

        if self.training:
            x1 = self.linear1(self.vgg(x1))
            x2 = self.linear2(x2)
            x2[:, 0, :] = x1 # replace start vectors with image representation
            x2 = torch.nn.utils.rnn.pack_padded_sequence(x2, lengths, batch_first=True)
            y_pred = self.linear3(torch.nn.utils.rnn.pad_packed_sequence(self.lstm(x2)[0], batch_first=True)[0])

            return torch.transpose(y_pred, 1, 2)

        else: # generate caption by sampling (feed generated words back into LSTM until end token reached)
            x2 = self.linear1(self.vgg(x1)).unsqueeze(0)
            endTokenVector = torch.zeros(1, 1, self.vocabSize, device=device, dtype=dtype)
            endTokenVector[0][0][self.endToken] = 1
            endTokenVector = self.linear2(endTokenVector)

            x_t = x2
            h_t = torch.zeros(1, 1, 512, device=device, dtype=dtype)
            c_t = h_t
            y_pred = []

            while not torch.equal(x_t, endTokenVector) and len(y_pred) < self.maxCapLength:
                (out, (h_t, c_t)) = self.lstm(x_t, (h_t, c_t))
                x_t = self.softmax(self.linear3(h_t))

                predictedWordIndex = torch.multinomial(x_t[0][0], 1)[0]
                x_t.new_zeros(x_t.size())
                x_t[0][0][predictedWordIndex] = 1
                x_t = self.linear2(x_t)

                y_pred.append(predictedWordIndex)

            return y_pred

trainDataType = 'train2014'
valDataType = 'val2014'

# retrieve images
images = get_images('cocoapi-master', trainDataType)
print('%d Training Images Loaded.' % len(images))
valImages = get_images('cocoapi-master', valDataType)
print('%d Validation Images Loaded.' % len(valImages))

# retrieve captions
captions = get_captions('cocoapi-master', trainDataType, images)
(vocab, vocabEncoding) = create_encoding(captions)
vocabSize = len(vocab)
valCaptions = get_captions('cocoapi-master', valDataType, valImages)

model = Captioner(vocabSize, vocabEncoding['<end>'])
if torch.cuda.is_available():
    model = model.cuda()

new_params = [] # parameters to be updated
for p in model.parameters():
    if p.requires_grad: # excludes CNN parameters
        new_params.append(p)

learningRate = 1e-3
numIters = 25
numBatches = 500
batchSize = len(images)//numBatches

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(new_params, lr=learningRate)
trainingLosses = torch.zeros(numIters*numBatches, device=device, dtype=dtype)
valLosses = torch.zeros(numIters, device=device, dtype=dtype)
maxBLEU = 0

for t in range(numIters):
    model.train()
    (images, captions) = shuffle(images, captions)
    (valImages, valCaptions) = shuffle(valImages, valCaptions)

    for b in range(numBatches):

        batchImages = get_image_data('..', trainDataType, images[b*batchSize:(b+1)*batchSize])
        batchCaptions = captions[b*batchSize:(b+1)*batchSize]
        (x1, x2, lengths, y) = prepare_data(batchImages, batchCaptions, vocabSize, vocabEncoding)

        y_pred = model(x1, x2, lengths)
        loss = loss_fn(y_pred, y)
        trainingLosses[t*numBatches+b] = loss

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Epoch %d Average Training Loss: %f" % (t + 1, torch.mean(trainingLosses[t*numBatches:(t+1)*numBatches]).item()))

    with torch.no_grad():

        valBatchImages = get_image_data('..', valDataType, valImages[0:batchSize])
        valBatchCaptions = valCaptions[0:batchSize]
        (x1_val, x2_val, lengths_val, y_val) = prepare_data(valBatchImages, valBatchCaptions, vocabSize, vocabEncoding)

        y_pred_val = model(x1_val, x2_val, lengths_val)
        loss_val = loss_fn(y_pred_val, y_val)
        valLosses[t] = loss_val

        print("Epoch %d Validation Loss: %f" % (t + 1, loss_val.item()))

        bleu = evaluate(valImages, valCaptions, model, valDataType, vocab)

        if bleu > maxBLEU:
            maxBLEU = bleu
            torch.save(model.state_dict(), 'captioner_model.pt')

torch.save(trainingLosses, 'training_loss.pt')
torch.save(valLosses, 'validation_loss.pt')
