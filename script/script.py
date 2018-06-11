#Initially forked from Bojan's kernel here: https://www.kaggle.com/tunguz/bow-meta-text-and-dense-features-lb-0-2242/code
#improvement using kernel from Nick Brook's kernel here: https://www.kaggle.com/nicapotato/bow-meta-text-and-dense-features-lgbm
#Used oof method from Faron's kernel here: https://www.kaggle.com/mmueller/stacking-starter?scriptVersionId=390867
#Used some text cleaning method from Muhammad Alfiansyah's kernel here: https://www.kaggle.com/muhammadalfiansyah/push-the-lgbm-v19
import time
notebookstart= time.time()

import os
import gc
import sys
import logging
import argparse
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm import tqdm

# Models Packages
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn import feature_selection
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import gensim

# Gradient Boosting
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.cross_validation import KFold
from sklearn.model_selection import KFold as KFOLD

# Tf-Idf
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from scipy.sparse import hstack, csr_matrix, issparse
from nltk.corpus import stopwords
import nltk.tokenize.toktok as toktok
from nltk.stem.snowball import SnowballStemmer 

# Viz
import re
import string

NFOLDS = 5
SEED = 42
VALID = False

parser = argparse.ArgumentParser()
parser.add_argument('--image_top', default=False)
parser.add_argument('--agg_feat', default=False)
parser.add_argument('--cluster', default=False)
parser.add_argument('--text', default=False)
parser.add_argument('--stem', default=False)
parser.add_argument('--ridge', default=False)
parser.add_argument('--tfidf', default=False)

args = parser.parse_args()

def cleanName(text):
    try:
        textProc = text.lower()
        # textProc = " ".join(map(str.strip, re.split('(\d+)',textProc)))
        #regex = re.compile(u'[^[:alpha:]]')
        #textProc = regex.sub(" ", textProc)
        textProc = re.sub('[!@#$_“”¨«»®´·º½¾¿¡§£₤‘’]', '', textProc)
        textProc = " ".join(textProc.split())
        return textProc
    except:
        return "name error"

def rmse(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power((y - y0), 2)))

##############################################################################################################
print("Selected Features:")
##############################################################################################################
print("image_top: {}".format(args.image_top))
print("agg_feat: {}".format(args.agg_feat))
print("cluster: {}".format(args.cluster))
print("text: {}".format(args.text))
print("stem: {}".format(args.stem))
print("ridge: {}".format(args.ridge))
print("tfidf: {}".format(args.tfidf))

##############################################################################################################
print("Data Load Stage")
##############################################################################################################
training = pd.read_csv('../input/avito-demand-prediction/train.csv', parse_dates = ["activation_date"])
testing = pd.read_csv('../input/avito-demand-prediction/test.csv', parse_dates = ["activation_date"])

ntrain = training.shape[0]
ntest = testing.shape[0]

y = training['deal_probability']
training.drop("deal_probability",axis=1, inplace=True)
print('Train shape: {} Rows, {} Columns'.format(*training.shape))
print('Test shape: {} Rows, {} Columns'.format(*testing.shape))
print("Combine Train and Test")
df = pd.concat([training,testing],axis=0)

df["price"] = np.log1p(df["price"])
df["image_top_1"].fillna(-999,inplace=True)
# df["week_of_year"] = df['activation_date'].dt.week
# df["day_of_month"] = df['activation_date'].dt.day
df["day_of_week"] = df['activation_date'].dt.weekday

df.drop(["activation_date","image"],axis=1,inplace=True)

# Aggregated Features
# https://www.kaggle.com/bminixhofer/aggregated-features-lightgbm
if args.agg_feat == 'True':
    ##############################################################################################################
    print("Aggregated Feature")
    ##############################################################################################################
    aggregated_features = pd.read_csv("../input/aggregated/aggregated_features.csv")
    df = df.merge(aggregated_features, on='user_id', how='left')
    df["avg_days_up_user"].fillna(-1, inplace=True)
    df["avg_times_up_user"].fillna(-1, inplace=True)

df.set_index('item_id', inplace=True)
training.set_index('item_id', inplace=True)
testing.set_index('item_id', inplace=True)
train_index = training.index
test_index = testing.index

# Predicted Image Top 1
if args.image_top == 'True':
    ##############################################################################################################
    print("Predicted Image Top 1 Feature")
    ##############################################################################################################
    training['image_top_1'] = pd.read_csv("../input/text2image-top-1/train_image_top_1_features.csv", index_col= "item_id")
    testing['image_top_1'] = pd.read_csv("../input/text2image-top-1/test_image_top_1_features.csv", index_col= "item_id")

if args.cluster == 'True':
    ##############################################################################################################
    print("Cluster Encoding Features")
    ##############################################################################################################
    from sklearn.cluster import DBSCAN
    agg_cols = ['region', 'city', 'parent_category_name',
                'category_name', 'user_type']
                # 'day_of_month','week_of_year']
    print("Clustering features: {}".format(agg_cols))

    def embed_category(dataframe, categories, target_category):
        group = dataframe[categories + [target_category]].groupby(categories)[target_category]
        hist = group.agg(lambda x: ' '.join(str(x)))
        group_index = hist.index
        sentences = [list(x) for x,_ in group]
        w2v = gensim.models.Word2Vec(sentences, min_count=1, size=500)
        return w2v, group_index, sentences
        
    def avg_w2v(Word2Vec, sentences):
        w2v_feature = []
        for sentence in tqdm(sentences):
            wv = 0
            sentlen = 0
            for word in sentence:
                try:
                    wv += Word2Vec.wv[word]
                    sentlen += 1
                except:
                    pass
            wv = wv/sentlen
            w2v_feature.append(wv)
        return np.asarray(w2v_feature)

    print("Building W2V")
    w2v, group_index, sentences = embed_category(df, agg_cols, "image_top_1")
    w2v_feature = avg_w2v(w2v, sentences)
    print("Running DBSCAN")
    db = DBSCAN(eps=0.3, min_samples=100, n_jobs=-1).fit(w2v_feature)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    complete_labels = [db.labels_[sentences.index(entry)] for entry in tqdm(df[agg_cols].values.tolist())]
    df['dbscan_cluster'] = pd.Series(complete_labels, index=df.index)
    df['dbscan_cluster'].fillna(-1, inplace=True)

##############################################################################################################
print("Target Encoding for Categorical Features")
##############################################################################################################
class TargetEncoder:
    # Adapted from https://www.kaggle.com/ogrellier/python-target-encoding-for-categorical-features
    def __repr__(self):
        return 'TargetEncoder'

    def __init__(self, cols, smoothing=1, min_samples_leaf=1, noise_level=0, keep_original=False):
        self.cols = cols
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf
        self.noise_level = noise_level
        self.keep_original = keep_original

    @staticmethod
    def add_noise(series, noise_level):
        return series * (1 + noise_level * np.random.randn(len(series)))

    def encode(self, train, test, target):
        for col in self.cols:
            if self.keep_original:
                train[col + '_te'], test[col + '_te'] = self.encode_column(train[col], test[col], target)
            else:
                train[col], test[col] = self.encode_column(train[col], test[col], target)
        return train, test

    def encode_column(self, trn_series, tst_series, target):
        temp = pd.concat([trn_series, target], axis=1)
        # Compute target mean
        averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
        # Compute smoothing
        smoothing = 1 / (1 + np.exp(-(averages["count"] - self.min_samples_leaf) / self.smoothing))
        # Apply average function to all target data
        prior = target.mean()
        # The bigger the count the less full_avg is taken into account
        averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
        averages.drop(['mean', 'count'], axis=1, inplace=True)
        # Apply averages to trn and tst series
        ft_trn_series = pd.merge(
            trn_series.to_frame(trn_series.name),
            averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
            on=trn_series.name,
            how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
        # pd.merge does not keep the index so restore it
        ft_trn_series.index = trn_series.index
        ft_tst_series = pd.merge(
            tst_series.to_frame(tst_series.name),
            averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
            on=tst_series.name,
            how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
        # pd.merge does not keep the index so restore it
        ft_tst_series.index = tst_series.index
        return self.add_noise(ft_trn_series, self.noise_level), self.add_noise(ft_tst_series, self.noise_level)

f_cats = ["region","city","parent_category_name","category_name","user_type","param_1","param_2","param_3"]
target_encode = TargetEncoder(min_samples_leaf=100, smoothing=10, noise_level=0.01,
                              keep_original=True, cols=f_cats)
training, testing = target_encode.encode(training, testing, y)

categorical = ["user_id","region","city","parent_category_name","category_name",
                "user_type","image_top_1","param_1","param_2","param_3"]
print("Start Label Encoding")
# Encoder:
lbl = preprocessing.LabelEncoder()
for col in categorical:
    df[col].fillna('Unknown')
    df[col] = lbl.fit_transform(df[col].astype(str))

if args.text == 'True':
    ##############################################################################################################
    print("Text Features")
    ##############################################################################################################
    textfeats = ["description", "title"]
    df['title'] = df['title'].apply(lambda x: cleanName(x))
    df["description"]   = df["description"].apply(lambda x: cleanName(x))
    df['desc_punc'] = df['description'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))

    for cols in textfeats:
        df[cols] = df[cols].astype(str)
        df[cols] = df[cols].astype(str).fillna('missing') # FILL NA
        df[cols] = df[cols].str.lower() # Lowercase all text, so that capitalized words dont get treated differently
        df[cols + '_num_char'] = df[cols].apply(lambda comment: len(str(comment)))
        df[cols + '_num_words'] = df[cols].apply(lambda comment: len(comment.split())) # Count number of Words
        df[cols + '_num_unique_words'] = df[cols].apply(lambda comment: len(set(w for w in comment.split())))
        df[cols + '_words_vs_unique'] = df[cols+'_num_unique_words'] / df[cols+'_num_words'] * 100 # Count Unique Words

    punct = set(string.punctuation)
    # print(punct)
    emoji = set()
    for s in df['title'].fillna('').astype(str):
        for c in s:
            if c.isdigit() or c.isalpha() or c.isalnum() or c.isspace() or c in punct:
                continue
            emoji.add(c)

    for s in df['description'].fillna('').astype(str):
        for c in str(s):
            if c.isdigit() or c.isalpha() or c.isalnum() or c.isspace() or c in punct:
                continue
            emoji.add(c)
    # print(''.join(emoji))
    # basic word and char stats for title
    df['n_titl_len'] = df['title'].fillna('').apply(len)
    df['n_titl_wds'] = df['title'].fillna('').apply(lambda x: len(x.split(' ')))
    df['n_titl_dig'] = df['title'].fillna('').apply(lambda x: sum(c.isdigit() for c in x))
    df['n_titl_cap'] = df['title'].fillna('').apply(lambda x: sum(c.isupper() for c in x))
    df['n_titl_spa'] = df['title'].fillna('').apply(lambda x: sum(c.isspace() for c in x))
    df['n_titl_pun'] = df['title'].fillna('').apply(lambda x: sum(c in punct for c in x))
    df['n_titl_emo'] = df['title'].fillna('').apply(lambda x: sum(c in emoji for c in x))

    # some ratio stats for title
    df['r_titl_wds'] = df['n_titl_wds']/(df['n_titl_len']+1)
    df['r_titl_dig'] = df['n_titl_dig']/(df['n_titl_len']+1)
    df['r_titl_cap'] = df['n_titl_cap']/(df['n_titl_len']+1)
    df['r_titl_spa'] = df['n_titl_spa']/(df['n_titl_len']+1)
    df['r_titl_pun'] = df['n_titl_pun']/(df['n_titl_len']+1)
    df['r_titl_emo'] = df['n_titl_emo']/(df['n_titl_len']+1)

    # basic word and char stats for description
    df['n_desc_len'] = df['description'].fillna('').apply(len)
    df['n_desc_wds'] = df['description'].fillna('').apply(lambda x: len(x.split(' ')))
    df['n_desc_dig'] = df['description'].fillna('').apply(lambda x: sum(c in punct for c in x))
    df['n_desc_cap'] = df['description'].fillna('').apply(lambda x: sum(c.isdigit() for c in x))
    df['n_desc_pun'] = df['description'].fillna('').apply(lambda x: sum(c.isupper() for c in x))
    df['n_desc_spa'] = df['description'].fillna('').apply(lambda x: sum(c.isspace() for c in x))
    df['n_desc_emo'] = df['description'].fillna('').apply(lambda x: sum(c in emoji for c in x))
    df['n_desc_row'] = df['description'].astype(str).apply(lambda x: x.count('/\n'))

    # some ratio stats
    df['r_desc_wds'] = (df['n_desc_wds']/(df['n_desc_len']+1))
    df['r_desc_dig'] = (df['n_desc_dig']/(df['n_desc_len']+1))
    df['r_desc_cap'] = (df['n_desc_cap']/(df['n_desc_len']+1))
    df['r_desc_spa'] = (df['n_desc_spa']/(df['n_desc_len']+1))
    df['r_desc_pun'] = (df['n_desc_pun']/(df['n_desc_len']+1))
    df['r_desc_row'] = (df['n_desc_row']/(df['n_desc_len']+1))
    df['r_desc_emo'] = (df['n_desc_emo']/(df['n_desc_len']+1))

    df['r_titl_des'] = (df['n_titl_len']/(df['n_desc_len']+1))

if args.tfidf == "True":
    ##############################################################################################################
    print("[TF-IDF] Term Frequency Inverse Document Frequency Stage")
    ##############################################################################################################
    if args.stem == 'True':
        print("With stemming")
        stemmer = SnowballStemmer("russian") 
        tokenizer = toktok.ToktokTokenizer()

        def stemRussian(word, stemmer):
            try:
                word.encode(encoding='utf-8').decode('ascii')
                return word
            except:
                return stemmer.stem(word)

        tqdm.pandas()
        if "stemmed_description.csv" not in os.listdir("."):
            df['description'] = df['description'].progress_apply(lambda x: " ".join([stemRussian(word, stemmer) for word in tokenizer.tokenize(x)]))
            df['description'].to_csv("stemmed_description.csv", index= False, header='description')
            df.to_pickle("stemmed")
        else:
            pass
            # df['description'] = pd.read_csv("stemmed_description.csv").values
            # read_pickle
    russian_stop = set(stopwords.words('russian'))
    tfidf_para = {
        "stop_words": russian_stop,
        "analyzer": 'word',
        "token_pattern": r'\w{1,}',
        "sublinear_tf": True,
        "dtype": np.float32,
        "norm": 'l2',
        #"min_df":5,
        #"max_df":.9,
        "smooth_idf":False
    }

    def get_col(col_name): return lambda x: x[col_name]
    ##I added to the max_features of the description. It did not change my score much but it may be worth investigating
    vectorizer = FeatureUnion([
            ('description',TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=17000,
                **tfidf_para,
                preprocessor=get_col('description'))),
            ('title',CountVectorizer(
                ngram_range=(1, 2),
                stop_words = russian_stop,
                #max_features=7000,
                preprocessor=get_col('title')))
        ])

    start_vect=time.time()

    #Fit my vectorizer on the entire dataset instead of the training rows
    #Score improved by .0001
    vectorizer.fit(df.to_dict('records'))
    ready_df = vectorizer.transform(df.to_dict('records'))
    print("TFIDF Feature Shape: {}".format(np.shape(ready_df)))
    tfvocab = vectorizer.get_feature_names()
    # tfvocab = svd_comp.columns
    print("Vectorization Runtime: %0.2f Minutes"%((time.time() - start_vect)/60))

# Drop Text Cols
textfeats = ["description", "title"]
# sys.exit(1)
df.drop(textfeats, axis=1,inplace=True)

if args.ridge == "True":
    ##############################################################################################################
    # https://www.kaggle.com/mmueller/stacking-starter?scriptVersionId=390867
    # help lightgbm converge faster
    print("Ridge Regression Features")
    ##############################################################################################################
    from sklearn.metrics import mean_squared_error
    from math import sqrt

    ridge_params = {'alpha':30.0, 'fit_intercept':True, 'normalize':False, 'copy_X':True,
                   'max_iter':None, 'tol':0.001, 'solver':'auto', 'random_state':SEED}
    kf = KFold(ntrain, n_folds=NFOLDS, shuffle=True, random_state=SEED)

    class SklearnWrapper(object):
        def __init__(self, clf, seed=0, params=None, seed_bool = True):
            if(seed_bool == True):
                params['random_state'] = seed
            self.clf = clf(**params)

        def train(self, x_train, y_train):
            self.clf.fit(x_train, y_train)

        def predict(self, x):
            return self.clf.predict(x)

    def get_oof(clf, x_train, y, x_test):
        oof_train = np.zeros((ntrain,))
        oof_test = np.zeros((ntest,))
        oof_test_skf = np.empty((NFOLDS, ntest))

        for i, (train_index, test_index) in enumerate(kf):
            print('Ridge Regression, Fold {}'.format(i))
            x_tr = x_train[train_index]
            y_tr = y[train_index]
            x_te = x_train[test_index]

            clf.train(x_tr, y_tr)

            oof_train[test_index] = clf.predict(x_te)
            oof_test_skf[i, :] = clf.predict(x_test)

        oof_test[:] = oof_test_skf.mean(axis=0)
        return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
        # return oof_test_skf

    ridge = SklearnWrapper(clf=Ridge, seed = SEED, params = ridge_params)
    ridge_oof_train, ridge_oof_test = get_oof(ridge, ready_df[:ntrain], y, ready_df[ntrain:])

    rms = sqrt(mean_squared_error(y, ridge_oof_train))
    print('Ridge OOF RMSE: {}'.format(rms))
    ridge_preds = np.concatenate([ridge_oof_train, ridge_oof_test])
    df['ridge_preds'] = ridge_preds

##############################################################################################################
print("Combine Dense Features with Sparse Text Bag of Words Features")
##############################################################################################################
if args.tfidf == "True":
    X = hstack([csr_matrix(df.loc[train_index,:].values),ready_df[0:train_index.shape[0]]]) # Sparse Matrix
    testing = hstack([csr_matrix(df.loc[test_index,:].values),ready_df[train_index.shape[0]:]])
    tfvocab = df.columns.tolist() + tfvocab
else:
    X = df.loc[train_index,:].values
    testing = df.loc[test_index,:].values
    tfvocab = df.columns.tolist()

for shape in [X,testing]:
    print("{} Rows and {} Cols".format(*shape.shape))
print("Feature Names Length: ".format(len(tfvocab)))
del df,vectorizer, ready_df
# del ridge_preds, ready_df
gc.collect();

##############################################################################################################
print("Modeling Stage")
##############################################################################################################
# Benchmark from Original Kernel
# train's rmse: 0.192423  valid's rmse: 0.218871

print("Light Gradient Boosting Regressor")
lgbm_params =  {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    # 'max_depth': 15,
    'num_leaves':450,
    'feature_fraction': 0.5,
    'bagging_fraction': 0.75,
    # 'min_data_in_leaf': 500,
    'bagging_freq': 50,
    'learning_rate': 0.001,
    'verbose': 0,
    'lambda_l1': 10,
    'lambda_l2': 10
}  

i = 0
nFolds = 10
cv_score = 0

models = []
temp_prediction = []
kf_ = KFOLD(n_splits=nFolds, shuffle=True, random_state=SEED)
for train, valid in kf_.split(X):
    if i == 0:    
        if issparse(X):
            X = X.tocsr()
        lgbtrain = lgb.Dataset(X[train], y[train],
                        feature_name=tfvocab,
                        categorical_feature = "")
        lgbvalid = lgb.Dataset(X[valid], y[valid],
                        feature_name=tfvocab,
                        categorical_feature = "")

        model = lgb.train(
            lgbm_params,
            lgbtrain,
            num_boost_round=20000,
            valid_sets=[lgbtrain, lgbvalid],
            valid_names=['train','valid'],
            early_stopping_rounds=50,
            verbose_eval=100
        )

        model.save_model('model_{}.txt'.format(i));i += 1
        validation_score = np.sqrt(metrics.mean_squared_error(y[valid], model.predict(X[valid])))
        print('Fold {}, RMSE: {}'.format(i,validation_score))
        cv_score += validation_score
        models.append(model)
    else:
        break

##############################################################################################################
print("Model Prediction Stage")
##############################################################################################################
model_prediction = 0
# model = lgb.Booster(model_file='model.txt')
for i,model in enumerate(models):
    print("Model {}".format(i))
    model_prediction = model_prediction + np.asarray(model.predict(testing))
model_prediction = model_prediction/len(models)
#Mixing lightgbm with ridge. I haven't really tested if this improves the score or not
#blend = 0.75*model_prediction + 0.25*ridge_oof_test[:,0]

model_submission = pd.DataFrame(model_prediction,columns=["deal_probability"],index=test_index)
model_submission['deal_probability'].clip(0.0, 1.0, inplace=True) # Between 0 and 1
model_submission.to_csv("submission_dart.csv",index=True,header=True)
# print("image_top: {},agg_feat: {}, mean_encoding: {},emoji: {},stem: {}".format(args.image_top,args.agg_feat,args.mean_encoding,args.emoji,args.stem))
print("Notebook Runtime: %0.2f Minutes"%((time.time() - notebookstart)/60))
