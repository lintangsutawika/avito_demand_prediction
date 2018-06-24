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

def rmse(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power((y - y0), 2)))

##############################################################################################################
print("Data Load Stage")
##############################################################################################################
training = pd.read_csv('../input/avito-demand-prediction/train.csv', parse_dates = ["activation_date"])
testing = pd.read_csv('../input/avito-demand-prediction/test.csv', parse_dates = ["activation_date"])

ntrain = training.shape[0]
ntest = testing.shape[0]

objective = 'multiclass'
metric = 'multi_logloss'

# training.drop("deal_probability",axis=1, inplace=True)
testing['deal_probability'] = -1
print('Train shape: {} Rows, {} Columns'.format(*training.shape))
print('Test shape: {} Rows, {} Columns'.format(*testing.shape))
print("Combine Train and Test")
df = pd.concat([training,testing],axis=0)
df.drop(["deal_probability"],axis=1,inplace=True)
df["price_full"] = df["price"]
df["price"] = np.log1p(df["price"]+0.01)
df["price"].fillna(-1, inplace=True)

# df["image_top_1"].fillna(-999,inplace=True)
# df["week_of_year"] = df['activation_date'].dt.week
# df["day_of_month"] = df['activation_date'].dt.day
df["day_of_week"] = df['activation_date'].dt.weekday

training["price_full"] = training["price"]
training["price"] = np.log1p(training["price"]+0.01)
training["price"].fillna(-1, inplace=True)
# training["image_top_1"].fillna(-999,inplace=True)
# training["week_of_year"] = training['activation_date'].dt.week
# training["day_of_month"] = training['activation_date'].dt.day
training["day_of_week"] = training['activation_date'].dt.weekday

testing["price_full"] = testing["price"]
testing["price"] = np.log1p(testing["price"]+0.01)
testing["price"].fillna(-1, inplace=True)
# testing["image_top_1"].fillna(-999,inplace=True)
# testing["week_of_year"] = testing['activation_date'].dt.week
# testing["day_of_month"] = testing['activation_date'].dt.day
testing["day_of_week"] = testing['activation_date'].dt.weekday

textfeats = ["description", "title"]
categorical = ["region","city","parent_category_name","category_name",
                "user_type","image_top_1","param_1","param_2","param_3","day_of_week"]
target_encode_category = categorical
df.drop(["activation_date"],axis=1,inplace=True)

##############################################################################################################
print("Image Confidence")
##############################################################################################################
image_confidence_train = pd.read_csv("../input/image-confidence/image_confidence_train.csv")
image_confidence_test = pd.read_csv("../input/image-confidence/image_confidence_test.csv")
image_confidence = pd.concat([image_confidence_train,image_confidence_test],axis=0)
df = df.merge(image_confidence, on='image', how='left')
df['image_confidence'].fillna(-1, inplace=True)
del image_confidence_train, image_confidence_test
gc.collect()

df.drop(["image"],axis=1,inplace=True)

target = 'param_1'

if target == 'param_1':
    df.set_index('item_id', inplace=True)
    train_index = df[df.param_1.notnull()].index
    test_index = df[df.param_1.isnull()].index
elif target == 'param_2':
    df.set_index('item_id', inplace=True)
    train_index = df[df.param_2.notnull()].index
    test_index = df[df.param_2.isnull()].index
elif target == 'param_3':
    df.set_index('item_id', inplace=True)
    train_index = df[df.param_3.notnull()].index
    test_index = df[df.param_3.isnull()].index

##############################################################################################################
print("Regular Encoding for Categorical Features")
##############################################################################################################
# print("Start Label Encoding")
# Encoder:
lbl = preprocessing.LabelEncoder()
for col in categorical:
    try:
        df[col].fillna('Unknown')
    except:
        pass
    df[col] = lbl.fit_transform(df[col].astype(str))


if target == 'param_1':
    y = df.param_1[train_index]
    df.drop(['param_1'],axis=1,inplace=True)
    categorical.remove('param_1')
elif target == 'param_2':
    y = df.param_2[train_index]
    df.drop(['param_2'],axis=1,inplace=True)
    categorical.remove('param_2')
elif target == 'param_3':
    y = df.param_3[train_index]
    df.drop(['param_3'],axis=1,inplace=True)
    categorical.remove('param_3')

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

    def encode(self, name, train, test, target):
        for col in self.cols:
            if self.keep_original:
                train[col + name], test[col + name] = self.encode_column(train[col], test[col], target)
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

f_cats = target_encode_category
te_cats = [cat+"_te_price_log" for cat in f_cats]
target_encode = TargetEncoder(min_samples_leaf=100, smoothing=10, noise_level=0.01,
                              keep_original=True, cols=f_cats)
# training, testing = target_encode.encode(training, testing, y)
training, testing = target_encode.encode("_te_price_log", training, testing, df['price'].iloc[:ntrain])
df = pd.concat([df,pd.concat([training[te_cats],testing[te_cats]],axis=0).set_index(df.index)], axis=1)

te_cats = [cat+"_te_price_full" for cat in f_cats]
target_encode = TargetEncoder(min_samples_leaf=100, smoothing=10, noise_level=0.01,
                              keep_original=True, cols=f_cats)
training, testing = target_encode.encode("_te_price_full", training, testing, df['price_full'].iloc[:ntrain])
df = pd.concat([df,pd.concat([training[te_cats],testing[te_cats]],axis=0).set_index(df.index)], axis=1)

te_cats = [cat+"_te_deal" for cat in f_cats]
target_encode = TargetEncoder(min_samples_leaf=100, smoothing=10, noise_level=0.01,
                              keep_original=True, cols=f_cats)
training, testing = target_encode.encode("_te_deal", training, testing, df['deal_probability'].iloc[:ntrain])
df = pd.concat([df,pd.concat([training[te_cats],testing[te_cats]],axis=0).set_index(df.index)], axis=1)

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
del temp
gc.collect()

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

##############################################################################################################
print("Statistical Encoding for Categorical Features")
############################################################################################################## 


df['avg_price_by_parent_category_name_user_type'] = df.groupby(['parent_category_name','user_type'])['price'].transform('mean')
df['std_price_by_parent_category_name_user_type'] = df.groupby(['parent_category_name','user_type'])['price'].transform('std')
df['var_price_by_parent_category_name_user_type'] = df.groupby(['parent_category_name','user_type'])['price'].transform('var')
df['min_price_by_parent_category_name_user_type'] = df.groupby(['parent_category_name','user_type'])['price'].transform('min')
df['q1_price_by_parent_category_name_user_type'] = df.groupby(['parent_category_name','user_type'])['price'].transform('quantile',q=0.25)
df['med_price_by_parent_category_name_user_type'] = df.groupby(['parent_category_name','user_type'])['price'].transform('quantile',q=0.5)
df['q3_price_by_parent_category_name_user_type'] = df.groupby(['parent_category_name','user_type'])['price'].transform('quantile',q=0.75)
df['max_price_by_parent_category_name_user_type'] = df.groupby(['parent_category_name','user_type'])['price'].transform('max')
df['distance_to_avg_price_by_parent_category_name_user_type'] = df['avg_price_by_parent_category_name_user_type'] - df['price']

df['avg_price_by_item_seq_number'] = df.groupby(['item_seq_number'])['price'].transform('mean')
df['std_price_by_item_seq_number'] = df.groupby(['item_seq_number'])['price'].transform('std')
df['var_price_by_item_seq_number'] = df.groupby(['item_seq_number'])['price'].transform('var')
df['min_price_by_item_seq_number'] = df.groupby(['item_seq_number'])['price'].transform('min')
df['q1_price_by_item_seq_number'] = df.groupby(['item_seq_number'])['price'].transform('quantile',q=0.25)
df['med_price_by_item_seq_number'] = df.groupby(['item_seq_number'])['price'].transform('quantile',q=0.5)
df['q3_price_by_item_seq_number'] = df.groupby(['item_seq_number'])['price'].transform('quantile',q=0.75)
df['max_price_by_item_seq_number'] = df.groupby(['item_seq_number'])['price'].transform('max')
df['distance_to_avg_price_by_item_seq_number'] = df['avg_price_by_item_seq_number'] - df['price']

df['avg_price_by_title_num_char'] = df.groupby(['title_num_char'])['price'].transform('mean')
df['std_price_by_title_num_char'] = df.groupby(['title_num_char'])['price'].transform('std')
df['var_price_by_title_num_char'] = df.groupby(['title_num_char'])['price'].transform('var')
df['min_price_by_title_num_char'] = df.groupby(['title_num_char'])['price'].transform('min')
df['q1_price_by_title_num_char'] = df.groupby(['title_num_char'])['price'].transform('quantile',q=0.25)
df['med_price_by_title_num_char'] = df.groupby(['title_num_char'])['price'].transform('quantile',q=0.5)
df['q3_price_by_title_num_char'] = df.groupby(['title_num_char'])['price'].transform('quantile',q=0.75)
df['max_price_by_title_num_char'] = df.groupby(['title_num_char'])['price'].transform('max')
df['distance_to_avg_price_by_title_num_char'] = df['avg_price_by_title_num_char'] - df['price']

df['avg_price_by_region_day_of_week'] = df.groupby(['region','day_of_week'])['price'].transform('mean')
df['std_price_by_region_day_of_week'] = df.groupby(['region','day_of_week'])['price'].transform('std')
df['var_price_by_region_day_of_week'] = df.groupby(['region','day_of_week'])['price'].transform('var')
df['min_price_by_region_day_of_week'] = df.groupby(['region','day_of_week'])['price'].transform('min')
df['q1_price_by_region_day_of_week'] = df.groupby(['region','day_of_week'])['price'].transform('quantile',q=0.25)
df['med_price_by_region_day_of_week'] = df.groupby(['region','day_of_week'])['price'].transform('quantile',q=0.5)
df['q3_price_by_region_day_of_week'] = df.groupby(['region','day_of_week'])['price'].transform('quantile',q=0.75)
df['max_price_by_region_day_of_week'] = df.groupby(['region','day_of_week'])['price'].transform('max')
df['distance_to_avg_price_by_region_day_of_week'] = df['avg_price_by_region_day_of_week'] - df['price']

df['avg_price_by_city'] = df.groupby(['city'])['price'].transform('mean')
df['std_price_by_city'] = df.groupby(['city'])['price'].transform('std')
df['var_price_by_city'] = df.groupby(['city'])['price'].transform('var')
df['min_price_by_city'] = df.groupby(['city'])['price'].transform('min')
df['q1_price_by_city'] = df.groupby(['city'])['price'].transform('quantile',q=0.25)
df['med_price_by_city'] = df.groupby(['city'])['price'].transform('quantile',q=0.5)
df['q3_price_by_city'] = df.groupby(['city'])['price'].transform('quantile',q=0.75)
df['max_price_by_city'] = df.groupby(['city'])['price'].transform('max')
df['distance_to_avg_price_by_city'] = df['avg_price_by_city'] - df['price']

df['avg_price_by_city_category_name'] = df.groupby(['city','category_name'])['price'].transform('mean')
df['std_price_by_city_category_name'] = df.groupby(['city','category_name'])['price'].transform('std')
df['var_price_by_city_category_name'] = df.groupby(['city','category_name'])['price'].transform('var')
df['min_price_by_city_category_name'] = df.groupby(['city','category_name'])['price'].transform('min')
df['q1_price_by_city_category_name'] = df.groupby(['city','category_name'])['price'].transform('quantile',q=0.25)
df['med_price_by_city_category_name'] = df.groupby(['city','category_name'])['price'].transform('quantile',q=0.5)
df['q3_price_by_city_category_name'] = df.groupby(['city','category_name'])['price'].transform('quantile',q=0.75)
df['max_price_by_city_category_name'] = df.groupby(['city','category_name'])['price'].transform('max')
df['distance_to_avg_price_by_city_category_name'] = df['avg_price_by_city_category_name'] - df['price']

df['avg_price_by_city_category_name_day_of_week'] = df.groupby(['city','category_name','day_of_week'])['price'].transform('mean')
df['std_price_by_city_category_name_day_of_week'] = df.groupby(['city','category_name','day_of_week'])['price'].transform('std')
df['var_price_by_city_category_name_day_of_week'] = df.groupby(['city','category_name','day_of_week'])['price'].transform('var')
df['min_price_by_city_category_name_day_of_week'] = df.groupby(['city','category_name','day_of_week'])['price'].transform('min')
df['q1_price_by_city_category_name_day_of_week'] = df.groupby(['city','category_name','day_of_week'])['price'].transform('quantile',q=0.25)
df['med_price_by_city_category_name_day_of_week'] = df.groupby(['city','category_name','day_of_week'])['price'].transform('quantile',q=0.5)
df['q3_price_by_city_category_name_day_of_week'] = df.groupby(['city','category_name','day_of_week'])['price'].transform('quantile',q=0.75)
df['max_price_by_city_category_name_day_of_week'] = df.groupby(['city','category_name','day_of_week'])['price'].transform('max')
df['distance_to_avg_price_by_city_category_name_day_of_week'] = df['avg_price_by_city_category_name_day_of_week'] - df['price']


##############################################################################################################
print("TFIDF Features")
##############################################################################################################
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
    stemmer = SnowballStemmer("russian") 
    tokenizer = toktok.ToktokTokenizer()
    df["title"] = df["title"].apply(lambda x: cleanName(x)) 
    df["description"] = df["description"].apply(lambda x: cleanName(x))
    df['description'] = df['description'].progress_apply(lambda x: " ".join([stemRussian(word, stemmer) for word in tokenizer.tokenize(x)]))
    df['description'].to_csv("stemmed_description.csv", index=True, header='description')
    df['title'] = df['title'].progress_apply(lambda x: " ".join([stemRussian(word, stemmer) for word in tokenizer.tokenize(x)]))
    df['title'].to_csv("stemmed_title.csv", index=True, header='title')
else:
    df.drop(textfeats,axis=1, inplace=True)
    stemmed_description = pd.read_csv("stemmed_description.csv", index_col='item_id').astype(str)
    stemmed_title = pd.read_csv("stemmed_title.csv", index_col='item_id').astype(str)
    df = pd.concat([df,stemmed_description], axis=1)
    df = pd.concat([df,stemmed_title], axis=1)
    
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
vectorizer.fit(df.to_dict('records'))
ready_df = vectorizer.transform(df.to_dict('records'))
print("TFIDF Feature Shape: {}".format(np.shape(ready_df)))
tfvocab = vectorizer.get_feature_names()
del vectorizer
gc.collect();


##############################################################################################################
# print("Use Binary")
##############################################################################################################
# binary_pred_train = pd.read_csv("binary_probability_train.csv", index_col='item_id').astype(float)
# binary_pred_test = pd.read_csv("binary_probability_test.csv", index_col='item_id').astype(float)
# df = pd.concat([df,pd.concat([binary_pred_train['binary_probability'],binary_pred_test['binary_probability']],axis=0)], axis=1)

##############################################################################################################
print("Build Dataset")
##############################################################################################################
#Drop selected features
df.drop(textfeats+["user_id"],axis=1, inplace=True)

temp_train = df.loc[train_index,:]
temp_test = df.loc[test_index,:]
tfvocab = df.columns.tolist() + tfvocab
del df
X = hstack([csr_matrix(temp_train.values),ready_df[0:train_index.shape[0]]]) # Sparse Matrix
testing = hstack([csr_matrix(temp_test.values),ready_df[train_index.shape[0]:]])
del ready_df
gc.collect();
X = X.tocsr()
testing = testing.tocsr()

for shape in [X,testing]:
    print("{} Rows and {} Cols".format(*shape.shape))
print("Feature Names Length: {}".format(len(tfvocab)))

##############################################################################################################
print("Modeling Stage")
##############################################################################################################

print("Light Gradient Boosting Regressor")
lgbm_params =  {
    'task': 'train',
    'boosting_type': 'gbdt',
    # 'objective': 'regression',
    # 'objective':'binary'
    # 'objective': 'poisson',
    'objective': 'multiclass',
    # 'metric': 'rmse',
    # 'metric': 'binary_logloss',
    'metric': 'multi_logloss',
    'num_class':len(np.unique(y))+1,
    # 'max_depth': 15,
    'num_leaves':50,
    'feature_fraction': 0.5,
    'bagging_fraction': 0.75,
    # 'min_data_in_leaf': 500,
    'bagging_freq': 5,
    'learning_rate': 0.001,
    'verbose': 0,
    'tree_learner':'voting',
    'lambda_l1': 10,
    'lambda_l2': 10,
    'max_bin': 100
}  

i = 0
nFolds = 5
cv_score = 0
models = []
temp_prediction = []
kf_ = KFOLD(n_splits=nFolds, shuffle=True, random_state=SEED)
for train, valid in kf_.split(X):
    # if issparse(X):
        # X = X.tocsr()
    lgbtrain = lgb.Dataset(X[train], y[train],
                    feature_name=tfvocab,
                    categorical_feature = categorical)
    lgbvalid = lgb.Dataset(X[valid], y[valid],
                    feature_name=tfvocab,
                    categorical_feature = categorical)

    model = lgb.train(
        lgbm_params,
        lgbtrain,
        num_boost_round=1,
        valid_sets=[lgbtrain, lgbvalid],
        valid_names=['train','valid'],
        learning_rates=lambda iter:0.1 * (0.999 ** iter),
        early_stopping_rounds=1,
        verbose_eval=1
    )

    model.save_model('model_{}.txt'.format(i));i += 1
    prediction = model.predict(X[valid])
    validation_score = np.sqrt(metrics.mean_squared_error(y[valid], model.predict(X[valid])))
    print('Fold {}, RMSE: {}'.format(i,validation_score))
    cv_score += validation_score
    models.append(model)
    feature = pd.DataFrame(data={'feature':model.feature_name(),'importance':model.feature_importance()})
        # print(feature.sort_values('importance'))

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
model_submission.round(5)
model_submission.to_csv("submission.csv",index=True,header=True)
# print("image_top: {},agg_feat: {}, mean_encoding: {},emoji: {},stem: {}".format(args.image_top,args.agg_feat,args.mean_encoding,args.emoji,args.stem))
print("Notebook Runtime: %0.2f Minutes"%((time.time() - notebookstart)/60))
