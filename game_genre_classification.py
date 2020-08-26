import pandas as pd
import numpy as np
import pymysql

from collections import Counter

import re
import jieba

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

def find_chinese(file):
    pattern = re.compile(r'[^\u4e00-\u9fa5]')
    chinese = re.sub(pattern, '', file)
    return chinese

def replace_puncs(x):
    return re.sub(r'[^\w\s]','',x)


def remove_blanks(x):
    return re.sub(r'\s{2,}','',x)

def cut_text(x):
    return list(jieba.cut(x,cut_all=False))


class TagSelector(BaseEstimator, TransformerMixin):

    def __init__(self,n_types = 20):
        self.n_types = n_types

    def fit(self,X):

        X = X.copy()
        count_tags = {k:v for k,v in
                          Counter([tag for tags in X.tags.apply(lambda x:x.split(',')).tolist()
         for tag in tags]).most_common(self.n_types)}

        self.tag_names = list(count_tags.keys())

        return self

    def transform(self,X):
        X = X.copy()

        all_tags = []
        for tags in X['tags'].apply(lambda x:x.split(',')).tolist():
            t = [tag for tag in tags if tag in self.tag_names]
            if len(t)>0:
                all_tags.append(t)
            else:
                all_tags.append(None)

        X['tags'] = all_tags
        X = X.dropna(subset = ['tags'])
        return X




class TagTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X):

        cnt_tags = Counter([tag for tags in X.tags.tolist()
                    for tag in tags]).most_common()

        game_types = [tag for tag, cnt in cnt_tags]

        self.gametype2id = {typename: i for i, typename in enumerate(game_types)}
        self.id2gametype = {v:k for k,v in self.gametype2id.items()}
        return self

    def transform(self, X):
        X = X.copy()
        X['encoded_tags'] = [[self.gametype2id[tag] for tag in tags] for tags in X.tags]
        return X




class TextCombiner(BaseEstimator, TransformerMixin):
    def fit(self,X):
        return self

    def transform(self,X):
        X['text'] = X['displayName'] + X['description'] + X['briefIntro']
        return X




class TextCleaner(BaseEstimator, TransformerMixin):

    def __init__(self, join_text = True):
        self.join_text = join_text

    def fit(self, X):
        return self

    def transform(self, X):
        X = X.dropna().copy()
        X['clean_text'] = X['text'].apply(remove_blanks).apply(replace_puncs).apply(find_chinese).apply(cut_text).values
        X = X[X['clean_text'].apply(lambda x:len(x) > 2)]
        if self.join_text:
            X['clean_text'] = X['clean_text'].apply(lambda x:' '.join(x))
        return X


class MultilabelEncoder(BaseEstimator, TransformerMixin):

    def fit(self, X):
        self.n_types = max([tag for tags in X.encoded_tags for tag in tags]) + 1
        return self

    def transform(self,X):
        eye = np.eye(self.n_types)

        vectors = []
        for tags in X.encoded_tags:
            vectors.append(eye[tags].sum(axis = 0))
        return np.array(vectors)


def train_model(n_tags = 40, max_features = 70000,test_size = 0.2):

    print('Reading data from database...')
    conn = pymysql.connect(host = '127.0.0.1', user = 'root', password = 'root123',db = 'game_info')

    df = pd.read_sql_query('''select displayName, description, briefIntro, tags
                                from test_game_info
                                where length(description) > 5 ''', conn)

    data_train,data_test = train_test_split(df, test_size = test_size, random_state = 42)

    print('Data preprocessing pipeline...')
    preprocess_pipe= Pipeline(steps = [('tag_selector',TagSelector(n_tags)),
                                      ('tag_transfer', TagTransformer()),
                                      ('text_combiner', TextCombiner()),
                                      ('text_cleaner', TextCleaner())])

    train_transer_data = preprocess_pipe.fit_transform(data_train)
    test_transfer_data = preprocess_pipe.transform(data_test)

    tfidf = TfidfVectorizer(ngram_range=(1,3),max_features = 70000)

    X_train = tfidf.fit_transform(train_transer_data['clean_text'])
    X_test = tfidf.transform(test_transfer_data['clean_text'])

    print('Encoding labels...')
    label_encoder = MultilabelEncoder()
    y_train = label_encoder.fit_transform(train_transer_data)
    y_test = label_encoder.transform(test_transfer_data)
    y_train[y_train >1] = 1
    y_test[y_test > 1] = 1

    model = OneVsRestClassifier(MultinomialNB())
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print('Accuracy score: {}'.format(accuracy))


    predictions = model.predict(X_test)
    for c in range(predictions.shape[1]):
        print(preprocess_pipe['tag_transfer'].id2gametype.get(c), ': ',roc_auc_score(y_test[:,c], predictions[:,c]))

    preds = model.predict_proba(X_test)

    test_preds = []
    for pred in preds:
        pred_tags = [preprocess_pipe['tag_transfer'].id2gametype.get(i) for i, p in enumerate(pred) if p>0.5]
        if len(pred_tags) == 0:
            pred_tags = [preprocess_pipe['tag_transfer'].id2gametype.get(p) for p in np.argsort(np.array(pred))[::-1][:3]]
        test_preds.append(pred_tags)

    views = test_transfer_data.copy()
    views['preds'] = test_preds


    while True:
        s = np.random.randint(0, len(preds))
        see = views.iloc[s:s+100][['displayName','text','preds']]
        print(see)
        input_ = input('view other predictions? (y/n)')
        if input_ == 'n':
            break

if __name__ == '__main__':
    train_model()
