import random
import os
import torch
import spacy
from torchtext.legacy import data
from torchtext.legacy import datasets
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import time
from torch.utils.tensorboard import SummaryWriter


'''
defined the Fields
loaded the dataset
created the splits
'''

print(os.getpid())

step1=[0]

SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
VOCAB_SIZE = 10000
LEARNING_RATE = 0.005
BATCH_SIZE = 64

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #rip intel, needed nvidia, its okay, will jsut train slow/inefficiently
PATH = 'SA-scratch-model2.pt'
nlp = spacy.load('en_core_web_sm')

#doing the data stuff https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/A%20-%20Using%20TorchText%20with%20Your%20Own%20Datasets.ipynb

torch.backends.cudnn.deterministic = True
#tokenizer = get_tokenizer('spacy')

TEXT = data.Field(tokenize='spacy', tokenizer_language='en_core_web_sm',
                  include_lengths=True)  # spacy splts on whitespace
LABEL = data.LabelField(dtype=torch.float)


train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
train_data, valid_data = train_data.split(random_state=random.seed(SEED))

print(len(train_data))
print(len(test_data))
print(vars(train_data[0]))

TEXT.build_vocab(train_data,
                 max_size = 25_000,
                 vectors="glove.6B.100d",
                 unk_init=torch.Tensor.normal_
                 )

LABEL.build_vocab(train_data)

print(TEXT.vocab.itos[:10])
print(LABEL.vocab.stoi)

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    sort_within_batch=True,
    device=DEVICE)


class RNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(
            input_dim, embedding_dim, padding_idx=pad_idx)

        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers,
                           bidirectional=bidirectional, dropout=dropout)

        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        embedded = self.dropout(self.embedding(text))

        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths.to('cpu'))

        #output, hidden = self.rnn(embedded)
        packed_output, (hidden, cell) = self.rnn(packed_embedded)

        output, output_lengths = nn.utils.rnn.pad_packed_sequence(
            packed_output)
        # hidden.squeeze_(0)
        #output = self.fc(hidden)
        hidden = self.dropout(
            torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        return self.fc(hidden)


INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
NUM_CLASSES = 1
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5
# https://stackoverflow.com/questions/61172400/what-does-padding-idx-do-in-nn-embeddings
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = RNN(len(TEXT.vocab),
            EMBEDDING_DIM,
            HIDDEN_DIM,
            OUTPUT_DIM,
            N_LAYERS,
            BIDIRECTIONAL,
            DROPOUT,
            PAD_IDX)


pretrained_embeddings = TEXT.vocab.vectors

print(pretrained_embeddings.shape)

model.embedding.weight.data.copy_(
    pretrained_embeddings)  # were not gonna go in blindly

UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

model.embedding.weight.data[UNK_IDX] = torch.zeros(
    EMBEDDING_DIM)  # I dont understand
model.embedding.weight.data[PAD_IDX] = torch.zeros(
    EMBEDDING_DIM)  # these lines

print(model.embedding.weight.data)

# -----------------------------Understood up to here----------------------------
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters())  # model.parameters() gives list?
print(f'model parameters: {model.parameters()}')
model = model.to(DEVICE)
criterion = criterion.to(DEVICE)


def binary_accuracy(preds, y):  # same as in the video but written in more lines
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    # round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc


# https://www.youtube.com/watch?v=VVDHU_TWwUg&ab_channel=PythonEngineer
def train(model, iterator, optimizer, criterion):
    # forward pass, backward pass, update weights
    model.train()
    epoch_acc = 0
    epoch_loss = 0
    for batch in iterator:

        optimizer.zero_grad()  # otherwise would accumulate gradient

        text, text_lengths = batch.text
        predictions = model(text, text_lengths).squeeze(
            1)  # does this get rid of the hidden value?
        #print(f'predictions: {predictions}')

        loss = criterion(predictions, batch.label)
        #print(f'loss: {loss}')

        acc = binary_accuracy(predictions, batch.label)
        print(f'accuracy: {acc}')
        step = step1[0]
        print('step: ',step)
        writer.add_scalar('Training Accuracy!!! ', acc, global_step = step)
        step1[0]+=1
        
        loss.backward()  # gradients dl/dw

        optimizer.step()  # update weights idk how it just good like that
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):

    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text
            predictions = model(text, text_lengths).squeeze(1)

            loss = criterion(predictions, batch.label)

            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def predict_sentiment(model, sentence):
    model.eval()
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    length = [len(indexed)]
    tensor = torch.LongTensor(indexed).to(DEVICE)
    tensor = tensor.unsqueeze(1)
    length_tensor = torch.LongTensor(length)
    prediction = torch.sigmoid(model(tensor, length_tensor))
    return prediction.item()


# https://www.datacamp.com/community/tutorials/tensorboard-tutorial
writer = SummaryWriter()

step = 0


best_valid_loss = float('inf')  # float inifinity?????!!!!!!!!!

model.load_state_dict(torch.load(PATH))
print('Model Loaded')

s=''
print('Train or Predict? (T/P)')
a = input()
if a == 'T':
    N_EPOCHS = int(input('epochs: '))
    print('Training...')
    for epoch in range(N_EPOCHS):
        print(epoch)
        start_time = time.time()
        train_loss, train_acc = train(
            model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
        end_time = time.time()
        writer.add_scalar('Training loss', train_loss, global_step = step)
        writer.add_scalar('Running loss', train_acc, global_step = step)
        step+=1
        if valid_loss < best_valid_loss:  # LESS VAlidation loss is pog idrk care abt over fitting at this point
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), PATH)
    '''    output = linear_model(sample, W, b)
		loss = (output - target) ** 2
		loss.backward()
		optimizer.step()'''
    print('FINISHED: ', 'epoch: ', epoch + 1, 'time: ', end_time - start_time, 'train loss: ',
          train_loss, ' train acc: ', train_acc, '| valid_loss :', valid_loss, ' valid: acc: ', valid_acc)
    print('Predicting sentiment for (this is garbage): ',
          predict_sentiment(model, 'this is garbage'))
    print('Predicting sentiment for (okay this is epic): ',
          predict_sentiment(model, 'okay, this is epic, I loved it'))
elif a == 'P':
    while True:
        s = input('input (type QUIT to quit): ')
        if s == 'QUIT':
            break
        print('Prediction: ', predict_sentiment(model, s))
else:
    print('ERROR')
