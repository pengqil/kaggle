import math
import numpy as np
import pandas as pd
import xgboost as xgb

from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split

print("Read Data and do preprocessing...")
stops = set(stopwords.words("english"))
stemmer = SnowballStemmer('english')


df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')

train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist()).astype(str)
test_qs = pd.Series(df_test['question1'].tolist() + df_test['question2'].tolist()).astype(str)


def preprocessing(row):
    row['q1'] = str(row['question1']).lower().split()
    row['q2'] = str(row['question2']).lower().split()


df_train = df_train.apply(preprocessing)
df_test = df_test.apply(preprocessing)


print("Feature Computation...")
def question_length_compare(row):
    return min(len(str(row['question1'])), len(str(row['question2']))) / \
           max(len(str(row['question1'])), len(str(row['question2'])))


def question_process(question, remove_punc=False, stem_words=False):
    word_dict = {}
    for word in question:
        if word not in stops:
            if remove_punc:
                word = ''.join([c for c in word if c not in punctuation])
            if stem_words:
                word = stemmer.stem(word)
            word_dict[word] = word_dict.get(word, 0) + 1
    return word_dict


def word_match_share(row):
    q1words, q2words = question_process(row['q1']), question_process(row['q2'])
    if len(q1words) == 0 or len(q2words) == 0:
        return 0
    shared_words_in_q1 = [w for w in q1words if w in q2words]
    shared_words_in_q2 = [w for w in q2words if w in q1words]
    return (len(shared_words_in_q1) + len(shared_words_in_q2)) / (len(q1words) + len(q2words))


def get_weight(count, min_count=2):
    if count < min_count:
        return 0
    else:
        return math.log(len(train_qs) / (count + 1))


print("Compute IDF")
words = [item for question in train_qs for item in set(str(question).lower().split())]
counts = Counter(words)
weights = {word: get_weight(count) for word, count in counts.items()}


def tfidf_cosine_distance(row):
    q1words, q2words = question_process(row['q1']), question_process(row['q2'])
    if len(q1words) == 0 or len(q2words) == 0:
        return 0
    q1_weights = [weights.get(w, 0) for w in q1words]
    q2_weights = [weights.get(w, 0) for w in q2words]

    shared_words = [w for w in q1words.keys() if w in q2words.keys()]
    shared_weights = [weights.get(w, 0) for w in shared_words]

    Rcosine_denominator = (np.sqrt(np.dot(q1_weights, q1_weights)) * np.sqrt(np.dot(q2_weights, q2_weights)))
    return np.dot(shared_weights, shared_weights) / Rcosine_denominator


def bigram_match(row):
    q1_2gram = set([i for i in zip(row['q1'], row['q1'][1:])])
    q2_2gram = set([i for i in zip(row['q2'], row['q2'][1:])])
    shared_2gram = q1_2gram.intersection(q2_2gram)
    if len(q1_2gram) + len(q2_2gram) == 0:
        R2gram = 0
    else:
        R2gram = len(shared_2gram) / (len(q1_2gram) + len(q2_2gram))
    return R2gram


def tfidf_word_match_share(row):
    q1words, q2words = question_process(row['q1']), question_process(row['q2'])
    if len(q1words) == 0 or len(q2words) == 0:
        return 0
    shared_weights = [weights.get(w, 0) * q1words[w] for w in q1words if w in q2words] + \
                     [weights.get(w, 0) * q2words[w] for w in q2words if w in q1words]
    total_weights = [weights.get(w, 0) * q1words[w] for w in q1words] + \
                    [weights.get(w, 0) * q2words[w] for w in q2words]
    return np.sum(shared_weights) / np.sum(total_weights)


def question_edit_distance(row):
    word1, word2 = row['q1'], row['q2']
    matrix = [[0] * (len(word2) + 1) for _ in range(len(word1) + 1)]
    for i in range(len(word2)):
        matrix[0][i + 1] = matrix[0][i] + 1
    for i in range(len(word1)):
        matrix[i + 1][0] = matrix[i][0] + 1
    for i in range(len(word1)):
        for j in range(len(word2)):
            matrix[i + 1][j + 1] = min(matrix[i][j + 1], matrix[i + 1][j]) + 1
            matrix[i + 1][j + 1] = min(matrix[i + 1][j + 1],
                                       matrix[i][j] + (0 if word1[i] == word2[j] else 1))
    return matrix[len(word1)][len(word2)] / max(len(word1), len(word2))


print("Compute Word2Vec")
word_data = [str(question).lower().split() for question in train_qs]
word2vec = Word2Vec(word_data, size=100, window=5, min_count=5, workers=4)
word2vec.save('model/word2vec.bin')


def cos_similarity(vector1, vector2):
    dotted = vector1.dot(vector2)
    vector1_norm = np.linalg.norm(vector1)
    vector2_norm = np.linalg.norm(vector2)
    vector_norms = np.multiply(vector1_norm, vector2_norm)
    neighbors = np.divide(dotted, vector_norms)
    return neighbors


def word2vec_similarity(row):
    question1 = np.array([word2vec.wv[word] * weights.get(word, 0)
                          for word in row['q1']
                          if word in word2vec.wv and word in weights])
    question2 = np.array([word2vec.wv[word] * weights.get(word, 0)
                          for word in row['q2']
                          if word in word2vec.wv and word in weights])
    if len(question1) == 0 or len(question2) == 0:
        return 0
    return cos_similarity(question1.mean(axis=0), question2.mean(axis=0))


def add_word_count(x, df, word):
    x['q1_' + word] = df['question1'].apply(lambda x: (word in str(x).lower())*1)
    x['q2_' + word] = df['question2'].apply(lambda x: (word in str(x).lower())*1)
    x[word + '_both'] = x['q1_' + word] * x['q2_' + word]


print("Compute Question Frequency")
df1 = df_train[['question1']].copy()
df2 = df_train[['question2']].copy()
df1_test = df_test[['question1']].copy()
df2_test = df_test[['question2']].copy()

df2.rename(columns={'question2': 'question1'}, inplace=True)
df2_test.rename(columns={'question2': 'question1'}, inplace=True)

train_questions = df1.append(df2)
train_questions = train_questions.append(df1_test)
train_questions = train_questions.append(df2_test)
train_questions.drop_duplicates(subset=['question1'], inplace=True)

train_questions.reset_index(inplace=True, drop=True)
questions_dict = pd.Series(train_questions.index.values,
                           index=train_questions.question1.values).to_dict()
train_cp = df_train.copy()
test_cp = df_test.copy()
train_cp.drop(['qid1', 'qid2'], axis=1, inplace=True)

test_cp['is_duplicate'] = -1
test_cp.rename(columns={'test_id': 'id'}, inplace=True)
comb = pd.concat([train_cp, test_cp])

comb['q1_hash'] = comb['question1'].map(questions_dict)
comb['q2_hash'] = comb['question2'].map(questions_dict)

q1_vc = comb.q1_hash.value_counts().to_dict()
q2_vc = comb.q2_hash.value_counts().to_dict()

def try_apply_dict(x,dict_to_apply):
    try:
        return dict_to_apply[x]
    except KeyError:
        return 0
#map to frequency space
comb['q1_freq'] = comb['q1_hash'].map(lambda x: try_apply_dict(x, q1_vc) + try_apply_dict(x, q2_vc))
comb['q2_freq'] = comb['q2_hash'].map(lambda x: try_apply_dict(x, q1_vc) + try_apply_dict(x, q2_vc))

train_comb = comb[comb['is_duplicate'] >= 0][['q1_freq', 'q2_freq']]
test_comb = comb[comb['is_duplicate'] < 0][['q1_freq', 'q2_freq']]

# Create training and testing data
x_train = pd.DataFrame()
x_test = pd.DataFrame()

print("Compute Training Data Features...")
x_train['word_match'] = df_train.apply(word_match_share, axis=1, raw=True)
x_train['tfidf_word_match'] = df_train.apply(tfidf_word_match_share, axis=1, raw=True)
x_train['question_length'] = df_train.apply(question_length_compare, axis=1, raw=True)
x_train['edit_distance'] = df_train.apply(question_edit_distance, axis=1, raw=True)
x_train['word2vec_distance'] = df_train.apply(word2vec_similarity, axis=1, raw=True)
x_train['sqrt_word_match'] = np.sqrt(x_train['word_match'])
x_train['bigram_match'] = df_train.apply(bigram_match, axis=1, raw=True)
x_train['tfidf_cosine_distance'] = df_train.apply(tfidf_cosine_distance, axis=1, raw=True)
x_train['q1_freq'] = train_comb['q1_freq']
x_train['q2_freq'] = train_comb['q2_freq']

add_word_count(x_train, df_train, 'how')
add_word_count(x_train, df_train, 'what')
add_word_count(x_train, df_train, 'which')
add_word_count(x_train, df_train, 'who')
add_word_count(x_train, df_train, 'where')
add_word_count(x_train, df_train, 'when')
add_word_count(x_train, df_train, 'why')


print("Compute Testing Data Features...")
x_test['word_match'] = df_test.apply(word_match_share, axis=1, raw=True)
x_test['tfidf_word_match'] = df_test.apply(tfidf_word_match_share, axis=1, raw=True)
x_test['question_length'] = df_test.apply(question_length_compare, axis=1, raw=True)
x_test['edit_distance'] = df_test.apply(question_edit_distance, axis=1, raw=True)
x_test['word2vec_distance'] = df_test.apply(word2vec_similarity, axis=1, raw=True)
x_test['sqrt_word_match'] = np.sqrt(x_test['word_match'])
x_test['bigram_match'] = df_test.apply(bigram_match, axis=1, raw=True)
x_test['tfidf_cosine_distance'] = df_test.apply(tfidf_cosine_distance, axis=1, raw=True)
x_test['q1_freq'] = test_comb['q1_freq']
x_test['q2_freq'] = test_comb['q2_freq']

add_word_count(x_test, df_test, 'how')
add_word_count(x_test, df_test, 'what')
add_word_count(x_test, df_test, 'which')
add_word_count(x_test, df_test, 'who')
add_word_count(x_test, df_test, 'where')
add_word_count(x_test, df_test, 'when')
add_word_count(x_test, df_test, 'why')

y_train = df_train['is_duplicate'].values


print("Sampling and validation split")
pos_train = x_train[y_train == 1]
neg_train = x_train[y_train == 0]

# Now we oversample the negative class
# There is likely a much more elegant way to do this...
p = 0.165
neg_needed = int(len(pos_train) / p) - (len(pos_train) + len(neg_train))
neg_sample = neg_train.sample(n = neg_needed, replace = True)
neg_train = pd.concat([neg_train, neg_sample])
print(len(pos_train) / (len(pos_train) + len(neg_train)))

x_train = pd.concat([pos_train, neg_train])
y_train = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist()
del pos_train, neg_train

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4242)

# Set our parameters for xgboost
print("Starts Training...")
params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'logloss'
params['eta'] = 0.11
params['max_depth'] = 5

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)

watchlist = [(d_train, 'train'), (d_valid, 'valid')]
bst = xgb.train(params, d_train, 600, watchlist, early_stopping_rounds=50, verbose_eval=10)

# Run Test
print("Running Test...")
d_test = xgb.DMatrix(x_test)
p_test = bst.predict(d_test)

sub = pd.DataFrame()
sub['test_id'] = df_test['test_id']
sub['is_duplicate'] = p_test
sub.to_csv('submission/ensemble_xgb.csv', index=False)