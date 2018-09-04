import torch
from torchvision import models
import random
import pickle
from nltk.translate.bleu_score import sentence_bleu
from nltk.corpus import stopwords
import sys
sys.path.insert(0, 'cocoapi-master/PythonAPI/')
from preprocess import get_images, get_image_data, get_captions, create_encoding

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
dtype = torch.float

def shuffle(captions, items):
    """
    Randomly shuffles captions and reorders items according to new caption order.

    Returns new list of items and captions.
    """

    for i in range(len(captions)): # keep track of which item index each caption corresponds to
        captions[i] = (captions[i], i)

    random.shuffle(captions)
    itemsOrdered = []

    for i in range(len(captions)):
        itemsOrdered.append(items[captions[i][1]]) # re-order items to match new caption order
        captions[i] = captions[i][0]

    return (captions, itemsOrdered)

def sort_descending_length(captions, items):
    """
    Sorts captions by decreasing length (number of words) and reorders items according to new caption order.

    Returns list of sorted captions and reordered items.
    """

    capsOrdered = []
    for i in range(len(captions)): # keep track of which image index each caption corresponds to
        capsOrdered.append(captions[i] + [i])

    capsOrdered.sort(reverse=True, key=len) # sort captions by length (descending)
    itemsOrdered = []

    for i in range(len(capsOrdered)):
        itemsOrdered.append(items[capsOrdered[i][-1]]) # re-order items to match new caption order
        capsOrdered[i] = capsOrdered[i][:-1]

    return (capsOrdered, itemsOrdered)

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
        lengths.append(len(caption) - 1)
        for w in range(len(caption)):
            if caption[w] in vocabEncoding:
                onehot[i][w][vocabEncoding[caption[w]]] = 1
                encoded[i][w] = vocabEncoding[caption[w]]
            else:  # unknown word (appears less than 5 times in training data)
                onehot[i][w][vocabEncoding['<unk>']] = 1
                encoded[i][w] = vocabEncoding['<unk>']

    return (onehot, encoded[:, 1:].long(), lengths)

def prepare_data(images, captions, vocab, vocabEncoding, initialCapWords=None):
    """
    Prepares input (x1, x2, lengths) and output (y) data from a batch of images (tensor) and captions (list).
    Also produces weights for weighted cross entropy loss.
    """

    for i in range(len(captions)):
        captions[i] = random.choice(captions[i])  # randomly choose one caption for each image

    weight = 5
    if initialCapWords == None:
        initialCapWords = [set()]*len(images)
        weight = 1

    (captions, items) = sort_descending_length(captions, [(images[i], initialCapWords[i]) for i in range(len(images))])
    initialCapWords = [item[1] for item in items]
    x1 = torch.zeros(images.size(), device=device, dtype=dtype)
    for i in range(len(captions)):
        x1[i] = items[i][0]

    (onehotCaps, encodedCaps, lengths) = encode(captions, len(vocab), vocabEncoding)
    weights = get_weights(encodedCaps, initialCapWords, lengths, weight, vocab)

    x2 = onehotCaps[:, :-1, :]
    y = onehotCaps[:, 1:]

    return (x1, x2, lengths, y, weights)

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
        y_pred = model(imageData[i].unsqueeze(0), beam=True, k=max(20, numCapsPerTest))

        for c in range(numCapsPerTest):
            cap = []
            for wordIndex in y_pred[c]:
                if vocab[wordIndex] not in {'<start>', '<end>', '<unk>', '<num>'}:
                    cap.append(vocab[wordIndex]) # decode words with vocab

            bleuScore = sentence_bleu(captions[i], cap)
            maxBLEU = max(maxBLEU, bleuScore)

        bleuTotal += maxBLEU

    print('BLEU-4 Score (%d %s Images): %f' % (numTests, dataType, bleuTotal/numTests))

    return bleuTotal/numTests

def get_initial_cap_words(images, dataType, initModel, file='', load=False, save=False):
    """
    Returns a list of sets, with each set containing the words that appeared in captions generated by initModel for a given image.
    """

    if load:
        caps_f = open(file, 'rb')
        capWords = pickle.load(caps_f)
        caps_f.close()
        return capWords

    initModel.eval()
    capWords = []

    for i in range(len(images)):
        img = get_image_data('..', dataType, [images[i]], blur=True)[0]
        y_pred = initModel(img.unsqueeze(0), beam=True, k=20)

        words = set()
        for c in range(5):
            words.update(set(y_pred[c]))

        capWords.append(words)

    if save:
        save_caps = open(file, 'wb')
        pickle.dump(capWords, save_caps)
        save_caps.close()

    return capWords

def get_weights(captions, initialCapWords, lengths, weight, vocab):
    """
    Generates weights for each word in each caption for weighted cross entropy loss based on whether or not the word was generated by the initial model.

    Arguments:
        captions - tensor of encoded captions
        initialCapWords - list of sets, with each set containing the words generated by the initial model for an image
        lengths - list of lengths for each caption
        weight - weight assigned to words that don't appear in initialCapWords
        vocab - list of words in caption vocabulary
    Returns:
        weights - tensor of weights with same shape as captions
    """

    stopWords = set(stopwords.words('english'))
    weights = torch.zeros(captions.size(), device=device, dtype=dtype)

    for i in range(len(captions)):
        for w in range(lengths[i]):
            if captions[i][w].item() in initialCapWords[i] or vocab[captions[i][w]] in stopWords:
                weights[i][w] = 1
            else:
                weights[i][w] = weight

    return weights

def weighted_ce_loss(outputs, labels, weights):
    """
    Custom loss function (weighted cross entropy loss).
    """
    lsoftmax = torch.nn.LogSoftmax(dim=2)

    batch_size = outputs.size()[0]
    seq_length = outputs.size()[1]
    loss = lsoftmax(outputs) * labels
    loss = torch.sum(loss, dim=2)
    loss = loss * weights

    return -torch.sum(loss) / batch_size / seq_length

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

        resnet = models.resnet152(pretrained=True)
        resnet = torch.nn.Sequential(*list(resnet.children())[:-1])

        for param in resnet.parameters():
            param.requires_grad = False

        self.resnet = resnet
        self.linear2 = torch.nn.Linear(self.vocabSize, 2048) # produces word embeddings
        self.lstm = torch.nn.LSTM(2048, 512, batch_first=True)
        self.linear3 = torch.nn.Linear(512, self.vocabSize)
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, x1, x2=torch.Tensor(), lengths=[], beam=True, k=1):
        """
        Implements the forward pass of the caption generator.

        Arguments:
            x1 - tensor representing image
            x2 - (if training) tensor representing sequence of words in caption, of shape (batch_size, max_seq_length, num_classes)
            lengths (if training) - lengths of captions represented by x2
            beam (if testing) - boolean; if True, beam search is performed when generating captions
            k (if beam is True) - beam size
        Returns:
            y_pred - (if training) tensor with sequence of probability distributions representing word predictions
                     (if beam search) list of lists of integers corresponding to predicted words for k best captions
                     (if sampling) list of integers corresponding to predicted words
        """

        self.resnet.eval()

        if self.training:
            x1 = self.resnet(x1).view(x1.size(0), -1)
            x2 = self.linear2(x2)
            x2[:, 0, :] = x1 # replace start vectors with image representation
            x2 = torch.nn.utils.rnn.pack_padded_sequence(x2, lengths, batch_first=True)
            y_pred = self.linear3(torch.nn.utils.rnn.pad_packed_sequence(self.lstm(x2)[0], batch_first=True)[0])

            return y_pred

        else: # generate caption(s)
            endTokenVector = torch.zeros(1, 1, self.vocabSize, device=device, dtype=dtype)
            endTokenVector[0][0][self.endToken] = 1
            endTokenVector = self.linear2(endTokenVector)

            x2 = self.resnet(x1).view(x1.size(0), -1).unsqueeze(0)
            x_t = x2
            h_t = torch.zeros(1, 1, 512, device=device, dtype=dtype)
            c_t = h_t

            if beam: # beam search

                startSeq = {'cap': [x_t], 'prob': 0, 'hidden': h_t, 'memory': c_t, 'y_pred': []}
                bestSeqs = [startSeq]
                allSeqs = []

                for c in range(self.maxCapLength):
                    for seq in bestSeqs:
                        if torch.equal(seq['cap'][-1], endTokenVector): # caption completed
                            allSeqs.append(seq)
                        else:
                            (out, (h_t, c_t)) = self.lstm(seq['cap'][-1], (seq['hidden'], seq['memory']))
                            x_t = self.softmax(self.linear3(h_t))
                            topK = torch.topk(x_t, k)
                            topK = (topK[0].squeeze(), topK[1].squeeze())

                            for i in range(k):
                                x_t = torch.zeros(1, 1, self.vocabSize, device=device, dtype=dtype)
                                x_t[0][0][topK[1][i]] = 1
                                x_t = self.linear2(x_t)

                                newSeq = {}
                                newSeq['cap'] = seq['cap'] + [x_t]
                                newSeq['prob'] = seq['prob'] + torch.log(topK[0][i])
                                newSeq['hidden'] = h_t
                                newSeq['memory'] = c_t
                                newSeq['y_pred'] = seq['y_pred'] + [topK[1][i].item()]

                                allSeqs.append(newSeq)

                    bestSeqs = sorted(allSeqs, reverse=True, key=lambda dict:dict['prob'])[:k] # keep k best sequences
                    allSeqs = []

                return [seq['y_pred'] for seq in bestSeqs]

            else: # sample one caption

                y_pred = []

                while not torch.equal(x_t, endTokenVector) and len(y_pred) < self.maxCapLength:
                    (out, (h_t, c_t)) = self.lstm(x_t, (h_t, c_t))
                    x_t = self.softmax(self.linear3(h_t))

                    predictedWordIndex = torch.multinomial(x_t[0][0], 1)[0]

                    x_t.new_zeros(x_t.size())
                    x_t[0][0][predictedWordIndex] = 1
                    x_t = self.linear2(x_t)

                    y_pred.append(predictedWordIndex.item())

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

initModel = Captioner(vocabSize, vocabEncoding['<end>'])
initModel.load_state_dict(torch.load('initial_captioner.pt'))
model = Captioner(vocabSize, vocabEncoding['<end>'])
if torch.cuda.is_available():
    initModel.cuda()
    model.cuda()

initialCaps = get_initial_cap_words(images, trainDataType, initModel, file='blurred_image_caps.pickle', load=True) # caption words generated from blurred images

new_params = [] # parameters to be updated
for p in model.parameters():
    if p.requires_grad: # excludes CNN parameters
        new_params.append(p)

numIters = 25
numBatches = 350
batchSize = len(images)//numBatches

optimizer = torch.optim.Adam(new_params)
trainingLosses = torch.zeros(numIters*numBatches, device=device, dtype=dtype)
valLosses = torch.zeros(numIters, device=device, dtype=dtype)
maxBLEU = 0

for t in range(numIters):
    model.train()
    (captions, items) = shuffle(captions, [(images[i], initialCaps[i]) for i in range(len(images))])
    images = [item[0] for item in items]
    initialCaps = [item[1] for item in items]
    (valCaptions, valImages) = shuffle(valCaptions, valImages)

    for b in range(numBatches):

        batchImages = get_image_data('..', trainDataType, images[b*batchSize:(b+1)*batchSize])
        batchCaptions = captions[b*batchSize:(b+1)*batchSize]
        initialCapWords = initialCaps[b*batchSize:(b+1)*batchSize]
        (x1, x2, lengths, y, weights) = prepare_data(batchImages, batchCaptions, vocab, vocabEncoding)

        y_pred = model(x1, x2, lengths)
        loss = weighted_ce_loss(y_pred, y, weights)
        trainingLosses[t*numBatches+b] = loss

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(new_params, 1)
        optimizer.step()

    print("Epoch %d Average Training Loss: %f" % (t + 1, torch.mean(trainingLosses[t*numBatches:(t+1)*numBatches]).item()))

    with torch.no_grad():

        valBatchImages = get_image_data('..', valDataType, valImages[0:batchSize])
        valBatchCaptions = valCaptions[0:batchSize]
        (x1_val, x2_val, lengths_val, y_val, weights) = prepare_data(valBatchImages, valBatchCaptions, vocab, vocabEncoding)

        y_pred_val = model(x1_val, x2_val, lengths_val)
        loss_val = weighted_ce_loss(y_pred_val, y_val, weights)
        valLosses[t] = loss_val

        print("Epoch %d Validation Loss: %f" % (t + 1, loss_val.item()))

        bleu = evaluate(valImages, valCaptions, model, valDataType, vocab, numTests=max(100, batchSize))

        if bleu > maxBLEU:
            maxBLEU = bleu
            torch.save(model.state_dict(), 'new_captioner_model.pt')

torch.save(trainingLosses, 'training_loss.pt')
torch.save(valLosses, 'validation_loss.pt')