import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from gensim.models import Word2Vec

stops = set(stopwords.words("english"))
stemmer = SnowballStemmer('english')

def text_to_wordlist(text, remove_stop_words=True, stem_words=False):
    # Clean the text, with the option to remove stop_words and to stem words.
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"what's", "", text)
    text = re.sub(r"What's", "", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"I'm", "I am", text)
    text = re.sub(r" m ", " am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\0k ", "0000 ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e-mail", "email", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"quikly", "quickly", text)
    text = re.sub(r" usa ", " America ", text)
    text = re.sub(r" USA ", " America ", text)
    text = re.sub(r" u s ", " America ", text)
    text = re.sub(r" uk ", " England ", text)
    text = re.sub(r" UK ", " England ", text)
    text = re.sub(r"india", "India", text)
    text = re.sub(r"china", "China", text)
    text = re.sub(r"chinese", "Chinese", text)
    text = re.sub(r"imrovement", "improvement", text)
    text = re.sub(r"intially", "initially", text)
    text = re.sub(r"quora", "Quora", text)
    text = re.sub(r" dms ", "direct messages ", text)
    text = re.sub(r"demonitization", "demonetization", text)
    text = re.sub(r"actived", "active", text)
    text = re.sub(r"kms", " kilometers ", text)
    text = re.sub(r"KMs", " kilometers ", text)
    text = re.sub(r" cs ", " computer science ", text)
    text = re.sub(r" upvotes ", " up votes ", text)
    text = re.sub(r" iPhone ", " phone ", text)
    text = re.sub(r"\0rs ", " rs ", text)
    text = re.sub(r"calender", "calendar", text)
    text = re.sub(r"ios", "operating system", text)
    text = re.sub(r"gps", "GPS", text)
    text = re.sub(r"gst", "GST", text)
    text = re.sub(r"programing", "programming", text)
    text = re.sub(r"bestfriend", "best friend", text)
    text = re.sub(r"dna", "DNA", text)
    text = re.sub(r"III", "3", text)
    text = re.sub(r"the US", "America", text)
    text = re.sub(r"Astrology", "astrology", text)
    text = re.sub(r"Method", "method", text)
    text = re.sub(r"Find", "find", text)
    text = re.sub(r"banglore", "Banglore", text)
    text = re.sub(r" J K ", " JK ", text)

    # Remove punctuation from text
    word_list = ''.join([c for c in text if c not in punctuation]).split()

    # Optionally, remove stop words
    if remove_stop_words:
        word_list = [w for w in word_list if w not in stops]

    # Optionally, shorten words to their stems
    if stem_words:
        word_list = [stemmer.stem(word) for word in word_list]

    # Return a list of words
    return " ".join(word_list)

# Text Cleaning
print('Processing text dataset')
df_train = pd.read_csv('data/train.csv')
df_train["q1"] = df_train["question1"].astype(str).apply(text_to_wordlist)
df_train["q2"] = df_train["question2"].astype(str).apply(text_to_wordlist)
df_test = pd.read_csv('data/test.csv')
df_test["q1"] = df_test["question1"].astype(str).apply(text_to_wordlist)
df_test["q2"] = df_test["question2"].astype(str).apply(text_to_wordlist)

MAX_NB_WORDS = 200000

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(df_train["q1"].tolist() + df_train["q2"].tolist() + df_test["q1"].tolist() + df_test["q2"].tolist())

df_train["sequence1"] = tokenizer.texts_to_sequences(df_train["q1"])
df_train["sequence2"] = tokenizer.texts_to_sequences(df_train["q2"])
df_test["sequence1"] = tokenizer.texts_to_sequences(df_test["q1"])
df_test["sequence2"] = tokenizer.texts_to_sequences(df_test["q2"])

df_train["sequence1"].head()

word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))

nb_words = min(MAX_NB_WORDS, len(word_index)) + 1
# print('Indexing word vectors')
EMBEDDING_DIM = 300
# word_data = [question.split() for question in (df_train["q1"].tolist() + df_train["q2"].tolist())]
# word2vec = Word2Vec(word_data, size=EMBEDDING_DIM, window=5, min_count=5, workers=4)
# word2vec.save('model/word2vec_lstm.bin')

MAX_SEQUENCE_LENGTH = 30
data_1 = pad_sequences(df_train["sequence1"], maxlen=MAX_SEQUENCE_LENGTH)
data_2 = pad_sequences(df_train["sequence2"], maxlen=MAX_SEQUENCE_LENGTH)
labels = np.array(df_train["is_duplicate"])
print('Shape of data tensor:', data_1.shape)
print('Shape of label tensor:', labels.shape)

data_1_test = pad_sequences(df_test["sequence1"], maxlen=MAX_SEQUENCE_LENGTH)
data_2_test = pad_sequences(df_test["sequence2"], maxlen=MAX_SEQUENCE_LENGTH)

# print('Preparing embedding matrix')
# embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
# for word, i in word_index.items():
#     if word in word2vec.wv:
#         embedding_matrix[i] = word2vec.wv[word]
# print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))


VALIDATION_SPLIT = 0.1
re_weight = True

perm = np.random.permutation(len(data_1))
idx_train = perm[:int(len(data_1) * (1 - VALIDATION_SPLIT))]
idx_val = perm[int(len(data_1) * (1 - VALIDATION_SPLIT)):]

data_1_train = np.vstack((data_1[idx_train], data_2[idx_train]))
data_2_train = np.vstack((data_2[idx_train], data_1[idx_train]))
labels_train = np.concatenate((labels[idx_train], labels[idx_train]))

data_1_val = np.vstack((data_1[idx_val], data_2[idx_val]))
data_2_val = np.vstack((data_2[idx_val], data_1[idx_val]))
labels_val = np.concatenate((labels[idx_val], labels[idx_val]))

weight_val = np.ones(len(labels_val))
if re_weight:
    weight_val *= 0.472001959
    weight_val[labels_val == 0] = 1.309028344

num_lstm = 225
num_dense = 125
rate_drop_lstm = 0.4
rate_drop_dense = 0.4
act = 'relu'

embedding_layer = Embedding(nb_words,
                            EMBEDDING_DIM,
                            # weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH)
                            # trainable=False)
lstm_layer = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)

sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences_1 = embedding_layer(sequence_1_input)
x1 = lstm_layer(embedded_sequences_1)

sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences_2 = embedding_layer(sequence_2_input)
y1 = lstm_layer(embedded_sequences_2)

merged = concatenate([x1, y1])
merged = Dropout(rate_drop_dense)(merged)
merged = BatchNormalization()(merged)

merged = Dense(num_dense, activation=act)(merged)
merged = Dropout(rate_drop_dense)(merged)
merged = BatchNormalization()(merged)

preds = Dense(1, activation='sigmoid')(merged)

model = Model(inputs=[sequence_1_input, sequence_2_input], outputs=preds)
model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])

if re_weight:
    class_weight = {0: 1.309028344, 1: 0.472001959}
else:
    class_weight = None

early_stopping = EarlyStopping(monitor='val_loss', patience=3)
bst_model_path = "model/lstm.h5"
model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

hist = model.fit([data_1_train, data_2_train], labels_train,
                 validation_data=([data_1_val, data_2_val], labels_val, weight_val),
                 epochs=200, batch_size=2048, shuffle=True,
                 class_weight=class_weight, callbacks=[early_stopping, model_checkpoint])

model.load_weights(bst_model_path)
bst_val_score = min(hist.history['val_loss'])

preds = model.predict([data_1_test, data_2_test], batch_size=8192, verbose=1)
preds += model.predict([data_2_test, data_1_test], batch_size=8192, verbose=1)
preds /= 2

submission = pd.DataFrame({'test_id': df_test["test_id"], 'is_duplicate': preds.ravel()})
submission.to_csv("submission/embedding_lstm.csv", index=False)

