# -----------------------------------------------------------------------------
# Action
# -----------------------------------------------------------------------------

'''
1. ----------------------------------------------------------------------------

Generate oligos (protein substrings of len k) to train on.

TODO: Which data?

- mycocosm
- vipr
- bakkis
'''

# cd /Users/phi/Dropbox/repos/nucleoform && conda activate shield
import numpy as np
import screed
from tqdm import tqdm

import torch
from torch import nn, optim

from nucleoform.data import partition, OligoDataset
from nucleoform.models import OligoRNN
from nucleoform.utils import frames, translate, dayhoff, windows


fp = 'data/coding.fna'  # predicted ORFs from prodigal on some genome
data = []

# Only the first one is a coding frame
labels = ['coding'] + 5 * ['non-coding']
wsize, overlap, truncate = 60, 30, True
# truncate .. discard windows with fewer than <wsize> characters
# TODO: Add padding

# TODO: subsample the useless proteins?
with screed.open(fp) as file:
    for i in tqdm(file):
        seqs = [dayhoff(translate(f)) for f in frames(i.sequence)]
        for label, seq in zip(labels, seqs):
            for w in windows(seq, wsize, overlap, truncate):
                data.append((label, w))
# In [8]: data[:10]
# Out[8]:
# [('coding', 'eddbfdececbbcccdbdeddbebebdfefcffebdcdcefcddbdfebbecfbdfeccc'),
#  ('coding', 'cffebdcdcefcddbdfebbecfbdfecccfebccccedfedcebbdbbdcbeecbcdbf'),

# TODO: We could turn these protein substrings into their UniRep representation
# as a form of transfer learning?


'''
2. ----------------------------------------------------------------------------

Put the data into a Dataloader object via the Dataset object, so we can use batch functions etc.

- https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
- dataset -> dataloader iterates over dataset
- https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset

Then split the data.
'''

train, dev, test = partition(OligoDataset(data))


'''
3. ----------------------------------------------------------------------------

Set up the neural net and choose params.


- https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
- https://github.com/udacity/deep-learning-v2-pytorch/blob/master/sentiment-rnn/Sentiment_RNN_Solution.ipynb
'''

# Params
all_letters = 'abcdef*'
# Because we reserve 0 for padding, we have one more number than letters.
n_letters = len(all_letters) + 1
n_output = 1  # we have 2 classes, 0 and 1, so we need only 1 number

# Hyperparams
n_batch = 64
n_hidden = 50
n_layers = 2
n_directions = 1  # 2 for BiLSTM -- not implemented yet
n_emb = 50  # dimension of embedding
lr = 3e-4  # Karpathy learning rate
epochs = 2
clip = 5 # gradient clipping
drop_prob = 0.1

# Logging
print_every = 1000


# Instantiate model, loss fn and optimizer
model = OligoRNN(n_letters, n_output, n_emb, n_hidden, n_layers, drop_prob)
criterion = nn.BCELoss()
# TODO: Use nn.BCEWithLogitsLoss(); hint: remove sigmoid fn from RNN
# https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html#bcewithlogitsloss
# > more numerically stable than using a plain Sigmoid followed by a BCELoss
# > It’s possible to trade off recall and precision by adding weights to positive examples
optimizer = optim.Adam(model.parameters(), lr=lr)


'''
4. ----------------------------------------------------------------------------

Train

TODO: To speed things up, we train on the dev set and validate on the test
set; on a GPU, train on train, validate on dev and test on test set.

TODO: Write test and inference loop.
'''


counter = 0
model.train()
# train for some number of epochs
for e in range(epochs):
    # initialize hidden state
    h = model.init_hidden(n_batch)

    # batch loop
    for inputs, labels in tqdm(train):
        counter += 1

        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history.
        h = tuple([each.data for each in h])

        # Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance.
        model.zero_grad()

        # Get the output from the model
        output, h = model(inputs, h)

        # Calculate the loss and perform backprop
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in 
        # RNNs / LSTMs.
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        # Adjust params
        optimizer.step()

        # Loss stats
        if counter % print_every == 0:
            # Get validation loss
            val_h = model.init_hidden(n_batch)
            val_losses = []
            model.eval()
            
            for inputs, labels in dev:
                # Creating new variables for the hidden state, otherwise
                # we'd backprop through the entire training history
                val_h = tuple([each.data for each in val_h])
                output, val_h = model(inputs, val_h)
                val_loss = criterion(output.squeeze(), labels.float())
                val_losses.append(val_loss.item())

            model.train()
            print("Epoch: {}/{}...".format(e+1, epochs),
                  "Step: {}...".format(counter),
                  "Loss: {:.6f}...".format(loss.item()),
                  "Val Loss: {:.6f}".format(np.mean(val_losses)))


# Test
model.eval()

val_losses = []
for inputs, labels in test:
    # Creating new variables for the hidden state, otherwise
    # we'd backprop through the entire training history
    val_h = tuple([each.data for each in val_h])
    output, val_h = model(inputs, val_h)
    val_loss = criterion(output.squeeze(), labels.float())
    val_losses.append(val_loss.item())
print("Epoch: {}/{}...".format(e+1, epochs),
      "Step: {}...".format(counter),
      "Loss: {:.6f}...".format(loss.item()),
      "Val Loss: {:.6f}".format(np.mean(val_losses)))


# Predict a single example
# https://discuss.pytorch.org/t/batch-prediction-for-a-model/12156/5
# model.batch_size = 1
# h = model.init_hidden(1)
# model(inputs[0:1], h)


'''
TODO: Test: Now the ultimate test is this: Feed it a genome assembly and go through the thing in "oligo steps"; mark coding or not. Correspondence of hits to prodigal? Any short ORFs detected? Viruses w/ overlapping frames.

TODO: I classify individual reading frames. This is wrong. We should classify which one of the six frames is coding (or none). In viruses (an bacteria?), two frames can be coding at the same time. So maybe a general protein recognizer is good.
'''
from nucleoform.transformations import numbersToLine, lineToNumbers
for ix, i in enumerate(labels):
    i = i.item()
    if i == 1:
        print(numbersToLine(inputs[ix]))
'''
eccbbebbbccebcbcbfeedcceefbbebccbbeafecbdebbbbbefeebbbebcccc
cebeefbbbbcecccdbbbeebbffbbdbbfdbcbfbeefcdebcbcbeefbbcbbdbbf
fdcdebdeedddceeccebccbeecccebbbfebdbebdbebecedcebbfecccbbbde
...
'''

'''
model.batch_size = 1
h = model.init_hidden(1)
# query = 'bbeacdeebbddebdbfdd*beebb*bdbbccbbbbbcbbedbdcbccbbddddbbbeae'
query = 'fbedebcebbbbbebebbdbfbbcbfdbcfbebbaedeebbbbebcbacbeeecccbccb'
x = lineToNumbers(query)
yhat, h_ = model(x.unsqueeze(0), h)  
# .unsqueeze() add the batch dimension to the start (we set batch_first=True)
round(yhat.squeeze().item())
# 0 or 1
'''

# Save model
# https://pytorch.org/tutorials/beginner/saving_loading_models.html


'''
from an oligo sequence, take first 120 nt, translate and choose, then translate protein and search in index
'''


torch.save(model.state_dict(), 'model.pt')  # a pickle object

model = OligoRNN(n_letters, n_output, n_emb, n_hidden, n_layers, drop_prob)
model.load_state_dict(torch.load('model.pt'))
# <All keys matched successfully>



# query = 'ATGGCTATCAAACTGCAGGACGGGAGTACACCTTGTCTGGCAGCTACACCTTCTGATCCACGCCCTACCGTGCTGGTGTTTGACTCCGGCGTCGGTGGGCTGTCGGTCTATGATGAGGTTCGGCATCTCCTGCCGGACCTTCATTACATCTACGCTTTCGATAACGTGGCATTCCCGTACGGAGAGAAGAGCGAAGACTTTATTGTCGAGCGCGTAGTGGAAATCGTCACGGCCGTACAACAACGCTACCCCCTGGCATTGGCAATTATTGCCTGTAATACGGCGAGTACTGTCTCTCTTCCTGCCCTGCGTGAGAAGTTTCCCTTCCCAGTGGTCGGCGTGGTCCCTGCGATTAAACCAGCGGCTCGTTTGACGGCCAATGGTGTGGTTGGCTTGCTTGCAACGCGCGGGACGGTAAAGCGTCCTTATACGCGTGAGCTGATTGAGCGCTTTGCCAATGAGTGCCAGATAGCCATGCTGGGTTCTGCCGAGCTGGTCGAAATCGCAGAAGCGAAGCTGCACGGTCAACCGGTGCCGCTGGAAGAGTTACAGCGCATATTGCGCCCGTGGCTGCGTATGGCTGAACCACCAGATACCGTTGTGTTGGGTTGTACTCATTTCCCGCTATTAAAAGATGAACTGCTTGCCGCGTTACCGGAAGGGACTCGCCTGGTGGACTCCGGAGCGGCCATTGCACGAAGAACAGCATGGTTACTTGAAAATGAAGCGCCAAATGCAAAATCTTCTGATGCGAATATCGCTTACTGCATGGCATTGACTGCAGAAACTGAGCAACTTTTACCCGTTTTACAACGTTATGGCTTCGAAACGCTCGAAAAACTGGCGCTATAG'

query = 'TGGCATTCCCGTACGGAGAGAAGAGCGAAGACTTTATTGTCGAGCGCGTAGTGGAAATCGTCACGGCCGTACAACAACGCTACCCCCTGGCATTGGCAATTATTGCCTGTAATACGGCGAGTACTGTCTCTCTTCCTGCCCTGCGTGAGAAGTTTCCCTTCCCAGTGGTCGGCGTGGTCCCTGCGATTAAACCAGCGGCTCGTTTGACGGCCAATGGTGTGGTTGGCTTGCTTGCAACGCGCGGGACGGTAAAGCG'

model.batch_size = 1
model.eval()
for f in frames(query):
    dh = dayhoff(translate(f))[:60]  # we trained on 60 aa
    x = lineToNumbers(dh)
    h = model.init_hidden(1)
    yhat, _ = model(x.unsqueeze(0), h)
    is_coding = round(yhat.squeeze().item())
    print(is_coding, dh)
    if is_coding == 1:
        aa = translate(f)

