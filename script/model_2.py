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

#

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

categorical = ["user_id","region","city","parent_category_name","category_name",
                "user_type","image_top_1","param_1","param_2","param_3","day_of_week"]

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

if args.categorical == "True":
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
    training, testing = target_encode.encode(training, testing, y)
    df = pd.concat([df,pd.concat([training[te_cats],testing[te_cats]],axis=0)], axis=1)
    
    # print("Start Label Encoding")
    # Encoder:
    lbl = preprocessing.LabelEncoder()
    for col in categorical:
        df[col].fillna('Unknown')
        df[col] = lbl.fit_transform(df[col].astype(str))
        
if args.cat2vec == 'True':
    from gensim.models import Word2Vec # categorical feature to vectors

if args.mean == "True":
    agg_cols = ['region', 'city', 'parent_category_name', 'category_name',
            'image_top_1', 'user_type','item_seq_number','day_of_week'];
    for c in tqdm(agg_cols):
        gp = tr.groupby(c)['deal_probability']
        mean = gp.mean()
        std  = gp.std()
        data[c + '_deal_probability_avg'] = data[c].map(mean)
        data[c + '_deal_probability_std'] = data[c].map(std)

    for c in tqdm(agg_cols):
        gp = tr.groupby(c)['price']
        mean = gp.mean()
        data[c + '_price_avg'] = data[c].map(mean)

if args.text == 'True':
    ##############################################################################################################
    print("Text Features")
    ##############################################################################################################
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

    textfeats = ["description", "title"]
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

df.drop(textfeats,axis=1, inplace=True)
if args.build_features == "True":
    sys.exit(1)

X = df.loc[train_index,:].values
testing = df.loc[test_index,:].values
tfvocab = df.columns.tolist()

for shape in [X,testing]:
    print("{} Rows and {} Cols".format(*shape.shape))
print("Feature Names Length: ".format(len(tfvocab)))

##############################################################################################################
print("Modeling Stage")
##############################################################################################################

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
        print(model.feature_importance())
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
