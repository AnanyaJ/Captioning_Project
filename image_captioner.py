import torch
from torchvision import models
import pickle

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
dtype = torch.float

def create_encoding(captions):
    """
    Maps each word present in captions to an integer. Returns the vocabulary size and this mapping.
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

    return len(vocab), vocabEncoding

def get_onehot(caption, vocabSize, vocabEncoding):
    """
    Creates and returns tensors representing caption by using vocabEncoding.

    Arguments:
        caption - a list of tokens (words) in an image caption
        vocabSize - number of words in the vocabulary
        vocabEncoding - mapping from words in vocabulary to integers
    Returns:
        capOneHot - a tensor of one-hot vectors, one for each token in caption, based on vocabEncoding
        capEncoded - a tensor of integers, one integer per token, based on vocabEncoding
    """
    capOneHot = torch.zeros(len(caption), vocabSize, device=device, dtype=dtype)
    capEncoded = torch.zeros(len(caption), device=device, dtype=dtype)

    for w in range(len(caption)):
        capOneHot[w][vocabEncoding[caption[w]]] = 1
        capEncoded[w] = vocabEncoding[caption[w]]

    return (capOneHot, capEncoded)

class Captioner(torch.nn.Module):

    def __init__(self, vocabSize):
        """
        Model for image caption generation with a CNN encoder to understand the image followed by an LSTM decoder to produce a corresponding caption.

        Arguments:
            vocabSize - the number of words in the vocabulary
        """
        super(Captioner, self).__init__()

        vgg = models.vgg16_bn(pretrained=True)
        newClassifier = torch.nn.Sequential(*list(vgg.classifier.children())[:-1]) # remove last FC layer
        vgg.classifier = newClassifier

        for param in vgg.parameters():  # freeze CNN parameters (except for last layer)
            param.requires_grad = False

        self.vgg = vgg
        self.linear1 = torch.nn.Linear(4096, 512) # new last layer of CNN
        self.lstm = torch.nn.LSTM(vocabSize, 512, batch_first=True)
        self.linear2 = torch.nn.Linear(512, vocabSize)

    def forward(self, x1, x2, training=True):
        """
        Implements the forward pass of the caption generator.

        Arguments:
            x1 - tensor representing image
            x2 - tensor representing sequence of words in caption
            training - boolean revealing whether this forward pass is for the purpose of training (if True) or testing
        Returns:
            y_pred - tensor with sequence of probability distributions representing word predictions
        """

        h_0 = self.linear1(self.vgg(x1)).unsqueeze(0) # image representation passed to LSTM as initial hidden state
        c_0 = torch.zeros(h_0.size(), device=device, dtype=dtype)

        if training:
            y_pred = self.linear2(self.lstm(x2, (h_0, c_0))[0])

            return torch.transpose(y_pred, 1, 2)

        else:
            # feed predicted words back into LSTM until end token is reached
            pass

# load images and captions
imgData = torch.load('img_data.pt')
captions_f = open('captions.pickle', 'rb')
captions = pickle.load(captions_f)
captions_f.close()

(vocabSize, vocabEncoding) = create_encoding(captions)

model = Captioner(vocabSize)
new_params = [] # parameters to be updated
for p in model.parameters():
    if p.requires_grad: # excludes CNN parameters
        new_params.append(p)

loss_fn = torch.nn.CrossEntropyLoss()
learningRate = 1e-4
optimizer = torch.optim.SGD(new_params, lr=learningRate)

for i in range(len(captions)):
    x1 = imgData[i].unsqueeze(0) # input to CNN (image)
    (x2, y) = get_onehot(captions[i][0], vocabSize, vocabEncoding) # use first caption for image
    x2 = x2[:-1].unsqueeze(0) # input to LSTM (caption)
    y = y[1:].unsqueeze(0).long()

    # forward pass
    y_pred = model(x1, x2)
    loss = loss_fn(y_pred, y)

    # backward pass (update parameters after every training example)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()




