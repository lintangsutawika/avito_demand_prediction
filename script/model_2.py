#Initially forked from Bojan's kernel here: https://www.kaggle.com/tunguz/bow-meta-text-and-dense-features-lb-0-2242/code
#improvement using kernel from Nick Brook's kernel here: https://www.kaggle.com/nicapotato/bow-meta-text-and-dense-features-lgbm
#Used oof method from Faron's kernel here: https://www.kaggle.com/mmueller/stacking-starter?scriptVersionId=390867
#Used some text cleaning method from Muhammad Alfiansyah's kernel here: https://www.kaggle.com/muhammadalfiansyah/push-the-lgbm-v19
import time
notebookstart= time.time()

import os
import gc
import sys
import argparse
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm

# Word Batch
import wordbatch
from wordbatch.extractors import WordBag
from wordbatch.models import FM_FTRL

# Cat2Vec
import copy
import gensim
from random import shuffle
from gensim.models import Word2Vec # categorical feature to vectors

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
parser.add_argument('--build_features', default=False)
parser.add_argument('--image_top', default=False)
parser.add_argument('--agg_feat', default=False)
parser.add_argument('--text', default=False)
parser.add_argument('--categorical', default=False)
parser.add_argument('--cat2vec', default=False)
parser.add_argument('--mean', default=False)
parser.add_argument('--target', default=False)
parser.add_argument('--wordbatch', default=False)
parser.add_argument('--image', default=False)
parser.add_argument('--sparse', default=False)
args = parser.parse_args()

def rmse(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power((y - y0), 2)))

##############################################################################################################
print("Selected Features:")
##############################################################################################################
print("build_features: {}".format(args.build_features))
print("image_top: {}".format(args.image_top))
print("agg_feat: {}".format(args.agg_feat))
print("text: {}".format(args.text))
print("categorical: {}".format(args.categorical))
print("cat2vec: {}".format(args.cat2vec))
print("mean: {}".format(args.mean))
print("target: {}".format(args.target))
print("wordbatch: {}".format(args.wordbatch))
print("image: {}".format(args.image))
print("sparse: {}".format(args.sparse))

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

# df["price"] = np.log1p(df["price"])
df["price"].fillna(-1, inplace=True)
# df["image_top_1"].fillna(-999,inplace=True)
# df["week_of_year"] = df['activation_date'].dt.week
# df["day_of_month"] = df['activation_date'].dt.day
df["day_of_week"] = df['activation_date'].dt.weekday

textfeats = ["description", "title"]
categorical = ["region","city","parent_category_name","category_name",
                "user_type","image_top_1","param_1","param_2","param_3","day_of_week"]

df.drop(["activation_date"],axis=1,inplace=True)

if args.image == 'True':
    ##############################################################################################################
    print("Image Confidence")
    ##############################################################################################################
    image_confidence_train = pd.read_csv("../input/image-confidence/image_confidence_train.csv")
    image_confidence_test = pd.read_csv("../input/image-confidence/image_confidence_test.csv")
    image_confidence = pd.concat([image_confidence_train,image_confidence_test],axis=0)
    df = df.merge(image_confidence, on='image', how='left')
    df['image_confidence'].fillna(-1, inplace=True)

df.drop(["image"],axis=1,inplace=True)

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
    df.drop(['image_top_1'], axis=1, inplace=True)
    df = pd.concat([df,pd.concat([training['image_top_1'],testing['image_top_1']],axis=0)], axis=1)

if args.mean == "True":
    ##############################################################################################################
    print("Statistical Encoding for Categorical Features")
    ############################################################################################################## 
    df['avg_price_by_param_1'] = df.groupby(['param_1'])['price'].transform('mean')
    df['std_price_by_param_1'] = df.groupby(['param_1'])['price'].transform('std')
    df['var_price_by_param_1'] = df.groupby(['param_1'])['price'].transform('var')
    df['med_price_by_param_1'] = df.groupby(['param_1'])['price'].transform('median')
    df['avg_price_by_city_param_1'] = df.groupby(['city','param_1'])['price'].transform('mean')
    df['std_price_by_city_param_1'] = df.groupby(['city','param_1'])['price'].transform('std')
    df['var_price_by_city_param_1'] = df.groupby(['city','param_1'])['price'].transform('var')
    df['med_price_by_city_param_1'] = df.groupby(['city','param_1'])['price'].transform('median')
    df['avg_price_by_city_image_top_1'] = df.groupby(['city','image_top_1'])['price'].transform('mean')
    df['std_price_by_city_image_top_1'] = df.groupby(['city','image_top_1'])['price'].transform('std')
    df['var_price_by_city_image_top_1'] = df.groupby(['city','image_top_1'])['price'].transform('var')
    df['med_price_by_city_image_top_1'] = df.groupby(['city','image_top_1'])['price'].transform('median')
    df['avg_price_by_city_image_top_1_day_of_week'] = df.groupby(['city','image_top_1','day_of_week'])['price'].transform('mean')
    df['std_price_by_city_image_top_1_day_of_week'] = df.groupby(['city','image_top_1','day_of_week'])['price'].transform('std')
    df['var_price_by_city_image_top_1_day_of_week'] = df.groupby(['city','image_top_1','day_of_week'])['price'].transform('var')
    df['med_price_by_city_image_top_1_day_of_week'] = df.groupby(['city','image_top_1','day_of_week'])['price'].transform('median')
    df['avg_price_by_city_category_name'] = df.groupby(['city','category_name'])['price'].transform('mean')
    df['std_price_by_city_category_name'] = df.groupby(['city','category_name'])['price'].transform('std')
    df['var_price_by_city_category_name'] = df.groupby(['city','category_name'])['price'].transform('var')
    df['med_price_by_city_category_name'] = df.groupby(['city','category_name'])['price'].transform('median')
    df['avg_price_by_city_category_name_day_of_week'] = df.groupby(['city','category_name','day_of_week'])['price'].transform('mean')
    df['std_price_by_city_category_name_day_of_week'] = df.groupby(['city','category_name','day_of_week'])['price'].transform('std')
    df['var_price_by_city_category_name_day_of_week'] = df.groupby(['city','category_name','day_of_week'])['price'].transform('var')
    df['med_price_by_city_category_name_day_of_week'] = df.groupby(['city','category_name','day_of_week'])['price'].transform('median')
    df['avg_image_top_1_by_city'] = df.groupby(['city'])['image_top_1'].transform('mean')
    df['std_image_top_1_by_city'] = df.groupby(['city'])['image_top_1'].transform('std')
    df['var_image_top_1_by_city'] = df.groupby(['city'])['image_top_1'].transform('var')
    df['med_image_top_1_by_city'] = df.groupby(['city'])['image_top_1'].transform('median')
    
if args.categorical == "True":    
    ##############################################################################################################
    print("Regular Encoding for Categorical Features")
    ##############################################################################################################
    # print("Start Label Encoding")
    # Encoder:
    lbl = preprocessing.LabelEncoder()
    for col in categorical:
        df[col].fillna('Unknown')
        df[col] = lbl.fit_transform(df[col].astype(str))
else:
    df.drop(categorical,axis=1, inplace=True)
    categorical = ""

if args.target == "True":
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

    f_cats = ["region","city","parent_category_name","category_name","user_type","param_1","param_2","param_3","image_top_1"]
    te_cats = [cat+"_te" for cat in f_cats]
    target_encode = TargetEncoder(min_samples_leaf=100, smoothing=10, noise_level=0.01,
                                  keep_original=True, cols=f_cats)
    # training, testing = target_encode.encode(training, testing, y)
    training, testing = target_encode.encode(training, testing, df['price'].iloc[:ntrain])
    df = pd.concat([df,pd.concat([training[te_cats],testing[te_cats]],axis=0)], axis=1)

if args.cat2vec == 'True':
    ##############################################################################################################
    print("Cat2Vec Encoding for Categorical Features")
    ##############################################################################################################
    # cat_cols = ['region', 'city', 'parent_category_name','category_name' 'user_type', 'user_id']
    cat_cols = categorical
    def apply_w2v(sentences, model, num_features):
        def _average_word_vectors(words, model, vocabulary, num_features):
            feature_vector = np.zeros((num_features,), dtype="float64")
            n_words = 0.
            for word in words:
                if word in vocabulary: 
                    n_words = n_words + 1.
                    feature_vector = np.add(feature_vector, model[word])

            if n_words:
                feature_vector = np.divide(feature_vector, n_words)
            return feature_vector
        
        vocab = set(model.wv.index2word)
        feats = [_average_word_vectors(s, model, vocab, num_features) for s in sentences]
        return np.array(feats)

    def gen_cat2vec_sentences(data):
        X_w2v = copy.deepcopy(data)
        names = list(X_w2v.columns.values)
        for c in names:
            X_w2v[c] = X_w2v[c].fillna('unknow').astype('category')
            X_w2v[c].cat.categories = ["%s %s" % (c,g) for g in X_w2v[c].cat.categories]
        X_w2v = X_w2v.values.tolist()
        return X_w2v

    def fit_cat2vec_model():
        X_w2v = gen_cat2vec_sentences(df.loc[:,cat_cols].sample(frac=0.6))
        for i in X_w2v:
            shuffle(i)
        model = Word2Vec(X_w2v, size=n_cat2vec_feature, window=n_cat2vec_window)
        return model

    n_cat2vec_feature  = len(cat_cols) # define the cat2vecs dimentions
    n_cat2vec_window   = len(cat_cols) * 2 # define the w2v window size

    c2v_model = fit_cat2vec_model()
    temp =pd.DataFrame(apply_w2v(gen_cat2vec_sentences(df.loc[:,cat_cols]), c2v_model, n_cat2vec_feature), 
                        columns=["cat2vec_"+element for element in cat_cols], index=df.index)
    df = pd.concat([df,temp], axis=1)

if args.text == 'True':
    ##############################################################################################################
    print("Text Features")
    ##############################################################################################################
    # df['title'] = df['title'].apply(lambda x: cleanName(x))
    # df["description"]   = df["description"].apply(lambda x: cleanName(x))
    df['desc_punc'] = df['description'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
    punct = set(string.punctuation)
    emoji = set()
    for cols in textfeats:
        for s in df[cols].fillna('').astype(str):
            for c in str(s):
                if c.isdigit() or c.isalpha() or c.isalnum() or c.isspace() or c in punct:
                    continue
                emoji.add(c)
        df[cols] = df[cols].astype(str)
        df[cols] = df[cols].astype(str).fillna('missing') # FILL NA
        df[cols + '_num_capital'] = df[cols].apply(lambda x: sum(c.isupper() for c in x))
        df[cols] = df[cols].str.lower() # Lowercase all text, so that capitalized words dont get treated differently
        df[cols + '_num_char'] = df[cols].apply(lambda x: len(str(x)))
        df[cols + '_num_words'] = df[cols].apply(lambda x: len(x.split())) # Count number of Words
        df[cols + '_num_digits'] = df[cols].apply(lambda x: sum(c.isdigit() for c in x))
        df[cols + '_num_spaces'] = df[cols].apply(lambda x: sum(c.isspace() for c in x))
        df[cols + '_num_punctuations'] = df[cols].apply(lambda x: sum(c in punct for c in x))
        df[cols + '_num_emoji'] = df[cols].apply(lambda x: sum(c in emoji for c in x))
        df[cols + '_num_unique_words'] = df[cols].apply(lambda x: len(set(w for w in x.split())))

        df[cols + '_digits_vs_char'] = df[cols + '_num_digits'] / df[cols + '_num_char'] * 100
        df[cols + '_capital_vs_char'] = df[cols + '_num_capital'] / df[cols + '_num_char'] * 100
        df[cols + '_spaces_vs_char'] = df[cols + '_num_spaces'] / df[cols + '_num_char'] * 100
        df[cols + '_punctuations_vs_char'] = df[cols + '_num_punctuations'] / df[cols + '_num_char'] * 100
        df[cols + '_emoji_vs_char'] = df[cols + '_num_emoji'] / df[cols + '_num_char'] * 100
        df[cols + '_words_vs_unique'] = df[cols+'_num_unique_words'] / df[cols+'_num_words'] * 100 # Count Unique Words

if args.wordbatch == 'True':
    ##############################################################################################################
    print("WordBatch Features")
    ##############################################################################################################
    from sklearn.metrics import mean_squared_error
    from math import sqrt

    kf = KFold(ntrain, n_folds=NFOLDS, shuffle=True, random_state=SEED)

    def normalize_text(text):
        text = text.lower().strip()
        for s in string.punctuation:
            text = text.replace(s, ' ')
        text = text.strip().split(' ')
        return u' '.join(x for x in text if len(x) > 1 and x not in stopwords)

    def cleanName(text):
        try:
            textProc = text.lower()
            # textProc = " ".join(map(str.strip, re.split('(\d+)',textProc)))
            #regex = re.compile(u'[^[:alpha:]]')
            #textProc = regex.sub(" ", textProc)
            textProc = re.sub(r"((\d+)[.,\-:]{0,}(\d+))","N",textProc)
            textProc = re.sub('[!@#$_“”¨«»®´·º½¾¿¡§£₤‘’]', '', textProc)
            textProc = " ".join(textProc.split())
            return textProc
        except:
            return "name error"

    def stemRussian(word, stemmer):
        try:
            word.encode(encoding='utf-8').decode('ascii')
            return word
        except:
            return stemmer.stem(word)

    tqdm.pandas()
    if "stemmed_description.csv" not in os.listdir("."):
        stemmer = SnowballStemmer("russian") 
        tokenizer = toktok.ToktokTokenizer()
        stopwords = {x: 1 for x in stopwords.words('russian')}
    
        df["title"] = df["title"].apply(lambda x: cleanName(x)) 
        df["description"] = df["description"].apply(lambda x: cleanName(x))
        df['description'] = df['description'].progress_apply(lambda x: " ".join([stemRussian(word, stemmer) for word in tokenizer.tokenize(x)]))
        df['description'].to_csv("stemmed_description.csv", index=True, header='description')
        df['title'] = df['title'].progress_apply(lambda x: " ".join([stemRussian(word, stemmer) for word in tokenizer.tokenize(x)]))
        df['title'].to_csv("stemmed_title.csv", index=True, header='title')
    else:
        df.drop(textfeats,axis=1, inplace=True)
        stemmed_description = pd.read_csv("stemmed_description.csv")
        stemmed_title = pd.read_csv("stemmed_title.csv")
        df = pd.concat([df,stemmed_description], axis=1)
        df = pd.concat([df,stemmed_title], axis=1)

    wb = wordbatch.WordBatch(normalize_text, extractor=(WordBag, {"hash_ngrams": 2,
                                                                  "hash_ngrams_weights": [1.5, 1.0],
                                                                  "hash_size": 2 ** 29,
                                                                  "norm": None,
                                                                  "tf": 'binary',
                                                                  "idf": None,
                                                                  }), procs=8)
    wb.dictionary_freeze = True
    X_title = wb.fit_transform(df['title'].fillna(''))
    del(wb)
    gc.collect()
    mask = np.where(X_title.getnnz(axis=0) > 3)[0]
    X_title = X_title[:, mask]
    print(X_title.shape)

    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_ind, test_ind) in enumerate(kf):
        print('Ridge Regression, Fold {}'.format(i))
        x_tr = X_title[:ntrain][train_ind]
        y_tr = y[train_ind]
        x_te = X_title[:ntrain][test_ind]

        model = Ridge(solver="sag", fit_intercept=True, random_state=205, alpha=3.3)
        model.fit(x_tr, y_tr)
        oof_train[test_ind] = model.predict(x_te)
        oof_test_skf[i, :] = model.predict(X_title[ntrain:])

    oof_test[:] = oof_test_skf.mean(axis=0)
    oof_train = oof_train.reshape(-1, 1)
    oof_test = oof_test.reshape(-1, 1)
    rms = sqrt(mean_squared_error(y, oof_train))
    print('Ridge OOF RMSE: {}'.format(rms))
    ridge_preds = np.concatenate([oof_train, oof_test])
    df['title_ridge_preds'] = ridge_preds
    gc.collect()

    wb = wordbatch.WordBatch(normalize_text, extractor=(WordBag, {"hash_ngrams": 2,
                                                                  "hash_ngrams_weights": [1.0, 1.0],
                                                                  "hash_size": 2 ** 28,
                                                                  "norm": "l2",
                                                                  "tf": 1.0,
                                                                  "idf": None}), procs=8)
    wb.dictionary_freeze = True
    X_description = wb.fit_transform(df['description'].fillna(''))
    del(wb)
    gc.collect()
    mask = np.where(X_description.getnnz(axis=0) > 8)[0]
    X_description = X_description[:, mask]
    print(X_description.shape)

    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_ind, test_ind) in enumerate(kf):
        print('Ridge Regression, Fold {}'.format(i))
        x_tr = X_description[:ntrain][train_ind]
        y_tr = y[train_ind]
        x_te = X_description[:ntrain][test_ind]

        model = Ridge(solver="sag", fit_intercept=True, random_state=205, alpha=3.3)
        model.fit(x_tr, y_tr)
        oof_train[test_ind] = model.predict(x_te)
        oof_test_skf[i, :] = model.predict(X_description[ntrain:])

    oof_test[:] = oof_test_skf.mean(axis=0)
    oof_train = oof_train.reshape(-1, 1)
    oof_test = oof_test.reshape(-1, 1)
    rms = sqrt(mean_squared_error(y, oof_train))
    print('Ridge OOF RMSE: {}'.format(rms))
    ridge_preds = np.concatenate([oof_train, oof_test])
    df['description_ridge_preds'] = ridge_preds
    gc.collect()

df.drop(textfeats+["user_id"],axis=1, inplace=True)
if args.build_features == "True":
    sys.exit(1)

X = df.loc[train_index,:].values
testing = df.loc[test_index,:].values
tfvocab = df.columns.tolist()

for shape in [X,testing]:
    print("{} Rows and {} Cols".format(*shape.shape))
print("Feature Names Length: {}".format(len(tfvocab)))

##############################################################################################################
print("Modeling Stage")
##############################################################################################################

print("Light Gradient Boosting Regressor")
lgbm_params =  {
    'task': 'train',
    'boosting_type': 'dart',
    'objective': 'regression',
    'metric': 'rmse',
    # 'max_depth': 15,
    'num_leaves':300,
    'feature_fraction': 0.5,
    'bagging_fraction': 0.75,
    # 'min_data_in_leaf': 500,
    'bagging_freq': 100,
    'learning_rate': 0.01,
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
                        categorical_feature = categorical)
        lgbvalid = lgb.Dataset(X[valid], y[valid],
                        feature_name=tfvocab,
                        categorical_feature = categorical)

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
        feature = pd.DataFrame(data={'feature':model.feature_name(),'importance':model.feature_importance()})
        # print(feature.sort_values('importance'))
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
model_submission.to_csv("submission.csv",index=True,header=True)
# print("image_top: {},agg_feat: {}, mean_encoding: {},emoji: {},stem: {}".format(args.image_top,args.agg_feat,args.mean_encoding,args.emoji,args.stem))
print("Notebook Runtime: %0.2f Minutes"%((time.time() - notebookstart)/60))
