#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
pd.set_option('display.max_colwidth', -1)
from sklearn.metrics import accuracy_score
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch.autograd import Variable
import torch.utils.data as data_utils
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
import csv
import pickle


# In[2]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[3]:


with open("train","r") as trainfile:
    corpus = trainfile.readlines()
train_corpus = [sentence for sentence in corpus]


# In[4]:


with open("dev","r") as devfile:
    corpus = devfile.readlines()
dev_corpus = [sentence for sentence in corpus]


# In[5]:


def getCorpus(data):
    cleaned_corpus = []
    sentence = []
    tags = []
    for words in data:
        if words[0]!='\n':
            word_tag = words.split(" ")
            sentence.append(word_tag[1])
            tags.append(word_tag[2].split('\n')[0])
        else:
            cleaned_corpus.append((sentence,tags))
            tags = []
            sentence = []
    cleaned_corpus.append((sentence,tags))
    return cleaned_corpus


# In[6]:


cleaned_train_corpus = getCorpus(train_corpus)
cleaned_dev_corpus = getCorpus(dev_corpus)


# In[7]:


df = pd.DataFrame.from_records(cleaned_train_corpus, columns =['Word', 'Tag'])
#df.tail(10)


# In[8]:


df_dev = pd.DataFrame.from_records(cleaned_dev_corpus, columns =['Word', 'Tag'])
#df_dev.head()


# In[9]:


#df.head(10)


# In[10]:


word_list = []
for sentence in df['Word']:
    for word in sentence:
        word_list.append(word)
vocab = set(word_list)


# In[11]:


tag_list = []
for tag_sentence in df['Tag']:
    #print(tag_sentence)
    for tag in tag_sentence:
        tag_list.append(tag)
tag_vocab = set(tag_list)


# In[12]:


with open("test","r") as testfile:
    corpus = testfile.readlines()
test_corpus = [sentence for sentence in corpus]


# In[13]:


cleaned_test_corpus = []
sentence = []
for words in test_corpus:
    if words[0]!='\n':
        word_tag = words.split(" ")
        sentence.append(word_tag[1].split('\n')[0])
    else:
        cleaned_test_corpus.append((sentence,0))
        sentence = []
cleaned_test_corpus.append((sentence,0))


# In[14]:


df_test = pd.DataFrame.from_records(cleaned_test_corpus, columns =['Word','Tag'])
#df_test = df_test.drop(columns=['Junk'])


# In[16]:


test_word_list = []
for sentence in df_test['Word']:
    for word in sentence:
        test_word_list.append(word)
test_vocab = set(test_word_list)
#len(test_vocab)


# In[17]:


word2index = {word: idx + 2 for idx, word in enumerate(vocab)}

word2index["--UNKNOWN_WORD--"]=0

word2index["--PADDING--"]=1

index2word = {idx: word for word, idx in word2index.items()}


# In[18]:


test_word = "Scotland"

test_word_idx = word2index[test_word]
test_word_lookup = index2word[test_word_idx]

#print("The index of the word {} is {}.".format(test_word, test_word_idx))
#print("The word with index {} is {}.".format(test_word_idx, test_word_lookup))


# In[19]:


tag2index = {tag: idx + 1 for idx, tag in enumerate(tag_vocab)}
tag2index["--PADDING--"] = 0

index2tag = {idx: word for word, idx in tag2index.items()}


# In[20]:


# with open('word2indexDict.pkl', 'wb') as file:
#     pickle.dump([word2index, index2word, tag2index, index2tag], file)


# In[21]:


with open('word2indexDict.pkl','rb') as f:
    word2index, index2word, tag2index, index2tag = pickle.load(f)


# In[22]:


#print(tag2index)
#print(word2index)


# In[23]:


#print(len(tag2index))
#print(len(word2index))


# In[24]:


def getDataset(dataframe,w2ix,testset=False):
    sentences = []
    for sentence in dataframe['Word']:
        #replace each token by its index if it is in vocab
        #else use index of UNK
        s = [w2ix[token] if token in vocab 
            else w2ix['--UNKNOWN_WORD--']
            for token in sentence]
        words = [ token for token in sentence]
        sentences.append((s,words))
    labels = []
    if testset:
        for tokenList in dataframe['Tag']:
            #print(tokenList)
            labels.append(tokenList)
        return sentences,labels
    else:

        for tokenList in dataframe['Tag']:
            #print(tokenList)
            s = [tag2index[token] for token in tokenList]
            labels.append(s)
        return sentences,labels


# In[25]:


train_sentences,train_labels = getDataset(df,word2index)


# In[26]:


dev_sentences,dev_labels = getDataset(df_dev,word2index)


# In[27]:


test_sentences,test_junk = getDataset(df_test,word2index,True)


# In[28]:


#print(dev_sentences[0])
#print(dev_labels[0])
#print(index2word[14279],index2word[11584])
#print(index2tag[2],index2tag[6])


# In[29]:


max_length_train = max(len(s[0]) for s in train_sentences)
#print(max_length_train)
#len(train_sentences)


# In[30]:


max_length_dev = max(len(s[0]) for s in dev_sentences)
#print(max_length_dev)


# In[31]:


max_length_test = max(len(s[0]) for s in test_sentences)
#print(max_length_test)


# In[32]:


padded_train_sentences = word2index['--PADDING--']*np.ones((len(train_sentences), max_length_train))
padded_train_tags = tag2index['--PADDING--']*np.ones((len(train_sentences), max_length_train))


# In[33]:


padded_dev_sentences = word2index['--PADDING--']*np.ones((len(dev_sentences), max_length_dev))
padded_dev_tags = tag2index['--PADDING--']*np.ones((len(dev_sentences), max_length_dev))


# In[34]:


padded_test_sentences = word2index['--PADDING--']*np.ones((len(test_sentences), max_length_test))
padded_test_junk = word2index['--PADDING--']*np.ones((len(test_sentences), max_length_test))


# In[35]:


for j in range(len(train_sentences)):
    cur_len = len(train_sentences[j][0])
    padded_train_sentences[j][:cur_len] = train_sentences[j][0]
    padded_train_tags[j][:cur_len] = train_labels[j]
# #since all data are indices, we convert them to torch LongTensors
# padded_train_sentences, padded_train_tags = torch.LongTensor(padded_train_sentences), torch.LongTensor(padded_train_tags)

# #convert Tensors to Variables
# padded_train_sentences, padded_train_tags = Variable(padded_train_sentences), Variable(padded_train_tags)


# In[36]:


for j in range(len(dev_sentences)):
    cur_len = len(dev_sentences[j][0])
    padded_dev_sentences[j][:cur_len] = dev_sentences[j][0]
    padded_dev_tags[j][:cur_len] = dev_labels[j]


# In[37]:


for j in range(len(test_sentences)):
    cur_len = len(test_sentences[j][0])
    padded_test_sentences[j][:cur_len] = test_sentences[j][0]
    padded_test_junk[j][:cur_len] = test_junk[j]


# In[38]:


padded_train_sentences = torch.tensor(padded_train_sentences).long()
padded_train_tags = torch.tensor(padded_train_tags).long()

padded_dev_sentences = torch.tensor(padded_dev_sentences).long()
padded_dev_tags = torch.tensor(padded_dev_tags).long()

padded_test_sentences = torch.tensor(padded_test_sentences).long()
padded_test_junk = torch.tensor(padded_test_junk).long()


# In[39]:


#TAG_COUNT = len(tag2index)


# In[40]:


# print(padded_train_sentences[0])
# print(padded_train_tags[0])

# print(padded_dev_sentences[0])
# print(padded_dev_tags[0])


# In[41]:


train = data_utils.TensorDataset(padded_train_sentences,padded_train_tags)
train_loader = data_utils.DataLoader(train, batch_size=32, shuffle=True)


# In[42]:


dev = data_utils.TensorDataset(padded_dev_sentences,padded_dev_tags)
dev_loader = data_utils.DataLoader(dev, batch_size=1, shuffle=False)


# In[43]:


test = data_utils.TensorDataset(padded_test_sentences,padded_test_junk)
test_loader = data_utils.DataLoader(test, batch_size=1, shuffle=False)


# In[44]:


class LSTM(nn.Module):
    def __init__(self, params):
        super(LSTM, self).__init__()

        #maps each token to an embedding_dim vector
        self.embedding = nn.Embedding(params["vocab_size"], params["embedding_dim"])

        #the LSTM takens embedded sentence
        self.lstm = nn.LSTM(params["embedding_dim"], params["lstm_hidden_dim"],bidirectional=True, batch_first=True)

        self.dropout = nn.Dropout(0.33)

        #fc layer transforms the output to give the final output layer
        self.fc1= nn.Linear(params["lstm_hidden_dim"]*2, params["linear_output_dim"])

        self.elu = nn.ELU()

        self.fc2= nn.Linear(params["linear_output_dim"], params["number_of_tags"])
        
    def forward(self, s):
        s = self.embedding(s)   # dim: batch_size x batch_max_len x embedding_dim

        #print(s.size()) #1x113x100
        #run the LSTM along the sentences of length batch_max_len
        s, _ = self.lstm(s)     # dim: batch_size x batch_max_len x lstm_hidden_dim                
        #print(s.size()) #1x113x256
        s = self.dropout(s)
        #reshape the Variable so that each row contains one token
        s = s.reshape(-1, s.shape[2])  # dim: batch_size*batch_max_len x lstm_hidden_dim
        #print(s.size()) #1*113x256
        #apply the fully connected layer and obtain the output for each token
        s = self.fc1(s)          # dim: batch_size*batch_max_len x num_tags
        #print(s.size()) #1*113x128
        s = self.elu(s)
        
        s = self.fc2(s)
        #print(s.size()) #1*113x10
        return F.log_softmax(s, dim=1)   # dim: batch_size*batch_max_len x num_tags


# In[45]:


#print(model)


# In[46]:


def loss_fn(outputs, labels):
    #reshape labels to give a flat vector of length batch_size*seq_len
    #labels = labels.view(-1)  

    #mask out 'PAD' tokens
    mask = (labels >= 1).float()

    #the number of tokens is the sum of elements in mask
    num_tokens = int(torch.sum(mask).item())

    #pick the values corresponding to labels and multiply by mask
    outputs = outputs[range(outputs.shape[0]), labels]*mask

    #cross entropy loss for all non 'PAD' tokens
    return -torch.sum(outputs)/num_tokens


# In[47]:


params = {"vocab_size":len(word2index),"embedding_dim":100,"lstm_hidden_dim":256,"linear_output_dim":128,"number_of_tags":len(tag2index)}
model = LSTM(params).to(device)
#criterion =  nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.98, nesterov=True, momentum = 0.9)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.9)
#scheduler = ExponentialLR(optimizer, gamma=0.9)
scheduler = ReduceLROnPlateau(optimizer, 'min', verbose = True)


# In[48]:


# model.train()
# for epoch in range(150):  # again, normally you would NOT do 300 epochs, it is toy data
#     epoch_loss = 0
#     epoch_acc = 0
#     for sentence, tags in train_loader:
#         sentence, tags = sentence.to(device), tags.to(device)
        
#         model.zero_grad()
        
#         tag_scores = model(sentence)
#         loss = loss_fn(tag_scores, tags.view(-1))
#         #acc = multi_acc(tag_scores, tags.view(-1))
        
#         loss.backward()
#         optimizer.step()
        
#         epoch_loss += loss.item()
#     scheduler.step(epoch_loss)
        
#     #print(f'Epoch {epoch+0:03}: | Loss: {epoch_loss/len(train_loader):.5f}')


# In[49]:


#torch.save(model, "blstm1.pt")


# In[50]:


model = torch.load("blstm1.pt")


# In[51]:


def convertNumbertoTag(data,i2x,pad):
    convertedData = []
    for sentence in data:
        for tag in sentence:
            if tag==pad:
                continue
            else:
                #print(tag)
                convertedData.append(i2x[tag])
        convertedData.append("\n")
    return convertedData


# In[52]:


y_pred_list = [] 
y_true_list = []
true_word_list = []
model.eval()
index = 0
with torch.no_grad():
    for X_batch, tags in dev_loader:
        X_batch = X_batch.to(device)
        #print(X_batch)
        #print(dev_sentences[index][1])
        y_test_pred = model(X_batch)
        #print(type(y_test_pred))
        _, y_pred_tags = torch.max(y_test_pred, dim = 1)
        #print(len(dev_sentences[index][1]))
        y_pred_list.append(y_pred_tags.cpu().numpy().tolist()[0:len(dev_sentences[index][1])])
        #print(y_pred_tags.cpu().numpy().tolist()[0:len(dev_sentences[index][1])])
        y_true_list.append(tags.squeeze(0).tolist()[:len(dev_sentences[index][1])])
        #print(tags.squeeze(0).tolist()[:len(dev_sentences[index][1])])
        for word in dev_sentences[index][1]:
            true_word_list.append(word)
        true_word_list.append("\n")
        index+=1


# In[53]:


#word_list = convertNumbertoTag(word_list,index2word,1)
y_true_list = convertNumbertoTag(y_true_list,index2tag,0)
y_pred_list = convertNumbertoTag(y_pred_list,index2tag,0)


# In[54]:


# len(y_pred_list)


# In[55]:


# print(len(y_true_list))


# In[56]:


# len(set(y_pred_list))


# In[57]:


acc = f1_score(y_true_list,y_pred_list, average = "macro")
# print(acc)


# In[58]:


# print(len(y_pred_list))
# print(len(true_word_list))


# In[59]:


index = 1
with open('dev1.out', 'w') as vocab_file:
    for i in range(len(true_word_list)):
        if true_word_list[i]=='\n':
            vocab_file.write("\n")
            index=1
        else:
            vocab_file.write(str(index)+" "+str(true_word_list[i])+" "+str(y_true_list[i])+" "+str(y_pred_list[i])+"\n")
            index+=1


# In[115]:


y_pred_list = [] 
y_true_list = []
true_word_list = []
model.eval()
index = 0
with torch.no_grad():
    for X_batch,tags in test_loader:
        X_batch = X_batch.to(device)
        #print(X_batch)
        #print(dev_sentences[index][1])
        y_test_pred = model(X_batch)
        #print(type(y_test_pred))
        _, y_pred_tags = torch.max(y_test_pred, dim = 1)
        #print(y_pred_tags.cpu().numpy().tolist())
        y_pred_list.append(y_pred_tags.cpu().numpy().tolist()[0:len(test_sentences[index][1])])
        #print(y_pred_tags.cpu().numpy().tolist())
        #print(test_sentences[index][1])
        for word in test_sentences[index][1]:
            true_word_list.append(word)
        true_word_list.append("\n")
        index+=1
        #y_true_list.append(tags.squeeze(0).tolist()[0:len(test_sentences[index][1])])


# In[116]:


#word_list = convertNumbertoTag(word_list,index2word,1)
#y_true_list = convertNumbertoTag(y_true_list,index2tag,0)
y_pred_list = convertNumbertoTag(y_pred_list,index2tag,0)


# In[117]:


# print(len(y_pred_list))
# print(len(true_word_list))


# In[118]:


index = 1
with open('test1.out', 'w') as vocab_file:
    for i in range(len(y_pred_list)):
        if true_word_list[i]=='\n':
            vocab_file.write("\n")
            index=1
        else:
            vocab_file.write(str(index)+" "+str(true_word_list[i])+" "+str(y_pred_list[i])+"\n")
            index+=1


# # Part 2
# 

# In[64]:


words = []
idx = 0
word2idxGlove = {}
vectors = []

with open('glove.6B.100d.txt', 'rb') as f:
    for l in f:
        line = l.decode().split()
        word = line[0]
        words.append(word)
        word2idxGlove[word] = idx
        idx += 1
        vect = np.array(line[1:]).astype(np.float)
        vectors.append(vect)


# In[ ]:


# print(vectors[word2idxGlove[words[0]]])


# In[65]:


glove = {w: vectors[word2idxGlove[w]] for w in words}


# In[66]:


# glove['the']


# In[67]:


dev_vocab = set(true_word_list)
# print(len(dev_vocab))
# print(len(vocab))
target_vocab = vocab.union(dev_vocab)
target_vocab = vocab.union(test_vocab)
#len(target_vocab)


# In[68]:


word2indexGlove = {word: idx + 2 for idx, word in enumerate(target_vocab)}

word2indexGlove["--UNKNOWN_WORD--"]=0

word2indexGlove["--PADDING--"]=1

index2wordGlove = {idx: word for word, idx in word2indexGlove.items()}

# print(index2wordGlove)


# In[69]:


# with open('word2indexDictGlove.pkl', 'wb') as file:
#     pickle.dump([word2indexGlove, index2wordGlove], file)


# In[70]:


with open('word2indexDictGlove.pkl','rb') as f:
    word2indexGlove, index2wordGlove = pickle.load(f)


# In[71]:


# print(index2wordGlove)


# In[72]:


t = list(index2wordGlove.values())


# In[73]:


#word2indexGlove['liberation']


# In[74]:


matrix_len = len(target_vocab) + 2
weights_matrix = np.zeros((matrix_len, 101))
words_found = 0
weights_matrix[0] = np.random.rand((101))
weights_matrix[1] = np.random.rand((101))
for i in range(len(t)-2):
    try: 
        if(t[i][0].isupper()):
            weights_matrix[i+2][:100] = glove[t[i].lower()]
            weights_matrix[i+2][100:] = 1.0
            words_found += 1
        else:
            weights_matrix[i+2][:100] = glove[t[i].lower()]
            weights_matrix[i+2][100:] = 0.0
            words_found += 1
    except KeyError:
        weights_matrix[i+2] = np.random.rand((101))
#print(words_found)


# In[ ]:


# len(weights_matrix[27318])


# In[75]:


# len(weights_matrix)


# In[76]:


# with open('weightsMatrix.pkl', 'wb') as file:
#     pickle.dump(weights_matrix, file)


# In[77]:


with open('weightsMatrix.pkl','rb') as f:
    weights_matrix = pickle.load(f)


# In[78]:


#print(words_found)
#print(word2indexGlove['hoping'])
# print(weights_matrix[6])
# glove['viva'.lower()]
#print(glove[index2wordGlove[83]])
#weights_matrix(Variable(torch.LongTensor([0])))


# In[79]:


train_sentences_Glove,train_labels_Glove = getDataset(df,word2indexGlove)


# In[80]:


dev_sentences_Glove,dev_labels_Glove = getDataset(df_dev,word2indexGlove)


# In[81]:


test_sentences_Glove,test_junk_Glove = getDataset(df_test,word2indexGlove,True)


# In[82]:


max_length_train_Glove = max(len(s[0]) for s in train_sentences_Glove)
# print(max_length_train_Glove)
len(train_sentences_Glove)
max_length_dev_Glove = max(len(s[0]) for s in dev_sentences)
# print(max_length_dev_Glove)


# In[83]:


padded_train_sentences_Glove = word2index['--PADDING--']*np.ones((len(train_sentences_Glove), max_length_train_Glove))
padded_train_tags_Glove = tag2index['--PADDING--']*np.ones((len(train_sentences_Glove), max_length_train_Glove))


# In[84]:


padded_dev_sentences_Glove = word2index['--PADDING--']*np.ones((len(dev_sentences_Glove), max_length_dev_Glove))
padded_dev_tags_Glove = tag2index['--PADDING--']*np.ones((len(dev_sentences_Glove), max_length_dev_Glove))


# In[85]:


padded_test_sentences_Glove = word2index['--PADDING--']*np.ones((len(test_sentences_Glove), max_length_test))
padded_test_junk_Glove = word2index['--PADDING--']*np.ones((len(test_sentences_Glove), max_length_test))


# In[86]:


for j in range(len(train_sentences_Glove)):
    cur_len = len(train_sentences_Glove[j][0])
    padded_train_sentences_Glove[j][:cur_len] = train_sentences_Glove[j][0]
    padded_train_tags_Glove[j][:cur_len] = train_labels_Glove[j]


# In[87]:


for j in range(len(dev_sentences_Glove)):
    cur_len = len(dev_sentences_Glove[j][0])
    padded_dev_sentences_Glove[j][:cur_len] = dev_sentences_Glove[j][0]
    padded_dev_tags_Glove[j][:cur_len] = dev_labels_Glove[j]


# In[88]:


for j in range(len(padded_test_sentences_Glove)):
    cur_len = len(test_sentences_Glove[j][0])
    padded_test_sentences_Glove[j][:cur_len] = test_sentences_Glove[j][0]
    padded_test_junk_Glove[j][:cur_len] = test_junk_Glove[j]


# In[89]:


padded_train_sentences_Glove = torch.tensor(padded_train_sentences_Glove).long()
padded_train_tags_Glove = torch.tensor(padded_train_tags_Glove).long()


# In[90]:


padded_dev_sentences_Glove = torch.tensor(padded_dev_sentences_Glove).long()
padded_dev_tags_Glove = torch.tensor(padded_dev_tags_Glove).long()


# In[91]:


padded_test_sentences_Glove = torch.tensor(padded_test_sentences_Glove).long()
padded_test_junk_Glove = torch.tensor(padded_test_junk_Glove).long()


# In[92]:


# print(padded_train_sentences_Glove[0])
# print(padded_train_tags_Glove[0])


# In[93]:


train_Glove = data_utils.TensorDataset(padded_train_sentences_Glove,padded_train_tags_Glove)
train_loader_Glove = data_utils.DataLoader(train_Glove, batch_size=32, shuffle=True)


# In[94]:


dev_Glove = data_utils.TensorDataset(padded_dev_sentences_Glove,padded_dev_tags_Glove)
dev_loader_Glove = data_utils.DataLoader(dev_Glove, batch_size=1, shuffle=False)


# In[95]:


test_Glove = data_utils.TensorDataset(padded_test_sentences_Glove,padded_test_junk_Glove)
test_loader_Glove = data_utils.DataLoader(test_Glove, batch_size=1, shuffle=False)


# In[96]:


def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.shape
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.weight.data.copy_(torch.from_numpy(weights_matrix))
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim


# In[97]:


class LSTM_Glove(nn.Module):
    def __init__(self, params):
        super(LSTM_Glove, self).__init__()

        #maps each token to an embedding_dim vector
        #self.embedding = nn.Embedding(params["vocab_size"], params["embedding_dim"])
        self.embedding, num_embeddings, embedding_dim = create_emb_layer(weights_matrix)
        
        #the LSTM takens embedded sentence
        self.lstm = nn.LSTM(params["embedding_dim"], params["lstm_hidden_dim"], bidirectional=True, batch_first=True)

        self.dropout = nn.Dropout(0.33)

        #fc layer transforms the output to give the final output layer
        self.fc1= nn.Linear(params["lstm_hidden_dim"]*2, params["linear_output_dim"])

        self.elu = nn.ELU()

        self.fc2= nn.Linear(params["linear_output_dim"], params["number_of_tags"])
        
    def forward(self, s):
        #print(s)
        s = self.embedding(s)   # dim: batch_size x batch_max_len x embedding_dim
        #print(s)
        #print(s.size()) #1x113x100
        #run the LSTM along the sentences of length batch_max_len
        s, _ = self.lstm(s)     # dim: batch_size x batch_max_len x lstm_hidden_dim                
        #print(s.size()) #1x113x256
        s = self.dropout(s)
        #reshape the Variable so that each row contains one token
        s = s.reshape(-1, s.shape[2])  # dim: batch_size*batch_max_len x lstm_hidden_dim
        #print(s.size()) #1*113x256
        #apply the fully connected layer and obtain the output for each token
        s = self.fc1(s)          # dim: batch_size*batch_max_len x num_tags
        #print(s.size()) #1*113x128
        s = self.elu(s)
        
        s = self.fc2(s)
        #print(s.size()) #1*113x10
        return F.log_softmax(s, dim=1)   # dim: batch_size*batch_max_len x num_tags


# In[98]:


def loss_fn(outputs, labels):
    #reshape labels to give a flat vector of length batch_size*seq_len
    #labels = labels.view(-1)  

    #mask out 'PAD' tokens
    mask = (labels >= 1).float()

    #the number of tokens is the sum of elements in mask
    num_tokens = int(torch.sum(mask).item())

    #pick the values corresponding to labels and multiply by mask
    outputs = outputs[range(outputs.shape[0]), labels]*mask
    
    #print(-torch.sum(outputs)/num_tokens)

    #cross entropy loss for all non 'PAD' tokens
    return -torch.sum(outputs)/num_tokens


# In[99]:


params = {"vocab_size":len(word2indexGlove),"embedding_dim":101,"lstm_hidden_dim":256,"linear_output_dim":128,"number_of_tags":len(tag2index)}
model_glove = LSTM_Glove(params).to(device)
#criterion =  nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_glove.parameters(), lr=0.1, nesterov=True, momentum = 0.9)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.9)
#scheduler = ExponentialLR(optimizer, gamma=0.9)
scheduler = ReduceLROnPlateau(optimizer, 'min', verbose = True)


# In[100]:


# model_glove.train()
# for epoch in range(200):  # again, normally you would NOT do 300 epochs, it is toy data
#     epoch_loss = 0
#     for sentence, tags in train_loader_Glove:
#         sentence, tags = sentence.to(device), tags.to(device)
        
#         model_glove.zero_grad()
        
#         tag_scores = model_glove(sentence)
#         #print(tag_scores.size())
#         #print(tags.view(-1).size())
#         #print(tag_scores,tags.view(-1))
#         loss = loss_fn(tag_scores, tags.view(-1))
#         #acc = multi_acc(tag_scores, tags.view(-1))
        
#         loss.backward()
#         optimizer.step()
        
#         epoch_loss += loss.item()
#     scheduler.step(epoch_loss/len(train_loader_Glove))
        
# #     print(f'Epoch {epoch+0:03}: | Loss: {epoch_loss/len(train_loader_Glove):.5f}')


# In[101]:


#torch.save(model_glove, "blstm2.pt")


# In[102]:


model_glove = torch.load("blstm2.pt")


# In[103]:


y_pred_list = [] 
y_true_list = []
true_word_list = []
model_glove.eval()
index = 0
with torch.no_grad():
    for X_batch, tags in dev_loader_Glove:
        X_batch = X_batch.to(device)
        #print(X_batch)
        #print(dev_sentences[index][1])
        y_test_pred = model_glove(X_batch)
        #print(y_test_pred.size())
        _, y_pred_tags = torch.max(y_test_pred, dim = 1)
        #print(y_pred_tags.size())
        #print(len(y_pred_tags.cpu().numpy()))
        #print(y_pred_tags.cpu().numpy().tolist())
        y_pred_list.append(y_pred_tags.cpu().numpy().tolist()[0:len(dev_sentences_Glove[index][1])])
        for word in dev_sentences_Glove[index][1]:
            true_word_list.append(word)
        true_word_list.append("\n")
        y_true_list.append(tags.squeeze(0).tolist()[0:len(dev_sentences_Glove[index][1])])
        index+=1


# In[104]:


y_true_list = convertNumbertoTag(y_true_list,index2tag,0)
y_pred_list = convertNumbertoTag(y_pred_list,index2tag,0)


# In[105]:


# print(len(y_pred_list))


# In[106]:


# len(set(y_true_list))


# In[107]:


# print(len(y_pred_list))
# print(len(true_word_list))


# In[108]:


acc = f1_score(y_true_list,y_pred_list, average = "macro")
# print(acc)


# In[109]:


index = 1
with open('dev2.out', 'w') as vocab_file:
    for i in range(len(true_word_list)):
        if true_word_list[i]=='\n':
            vocab_file.write("\n")
            index=1
        else:
            vocab_file.write(str(index)+" "+str(true_word_list[i])+" "+str(y_true_list[i])+" "+str(y_pred_list[i])+"\n")
            index+=1


# In[110]:


y_pred_list = [] 
y_true_list = []
true_word_list = []
model_glove.eval()
index = 0
with torch.no_grad():
    for X_batch, tags in test_loader_Glove:
        X_batch = X_batch.to(device)
        #print(X_batch)
        #print(dev_sentences[index][1])
        y_test_pred = model_glove(X_batch)
        #print(y_test_pred.size())
        _, y_pred_tags = torch.max(y_test_pred, dim = 1)
        #print(y_pred_tags.size())
        #print(len(y_pred_tags.cpu().numpy()))
        #print(y_pred_tags.cpu().numpy().tolist())
        y_pred_list.append(y_pred_tags.cpu().numpy().tolist()[0:len(test_sentences_Glove[index][1])])
        for word in test_sentences_Glove[index][1]:
            true_word_list.append(word)
        true_word_list.append("\n")
        #y_true_list.append(tags.squeeze(0).tolist()[0:len(dev_sentences_Glove[index][1])])
        index+=1


# In[111]:


#word_list = convertNumbertoTag(word_list,index2word,1)
#y_true_list = convertNumbertoTag(y_true_list,index2tag,0)
y_pred_list = convertNumbertoTag(y_pred_list,index2tag,0)


# In[112]:


# print(len(y_pred_list))
# print(len(true_word_list))


# In[113]:


index = 1
with open('test2.out', 'w') as vocab_file:
    for i in range(len(y_pred_list)):
        if true_word_list[i]=='\n':
            vocab_file.write("\n")
            index=1
        else:
            vocab_file.write(str(index)+" "+str(true_word_list[i])+" "+str(y_pred_list[i])+"\n")
            index+=1


# In[ ]:




