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
parser.add_argument('--deal', default=False)
parser.add_argument('--compare', default=False)
parser.add_argument('--tfidf', default=False)
parser.add_argument('--test', default=False)
parser.add_argument('--binary', default=False)
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
print("deal: {}".format(args.deal))
print("compare: {}".format(args.compare))
print("tfidf: {}".format(args.tfidf))
print("test: {}".format(args.test))
print("binary: {}".format(args.binary))

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

if args.image == 'True':
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
    del aggregated_features
    gc.collect()

if args.categorical == "True":    
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
else:
    df.drop(categorical,axis=1, inplace=True)
    categorical = ""


target = 'param_1'

if target == 'param_1':
    df.set_index('item_id', inplace=True)
    train_index = df.param_1.notnull().index
    test_index = df.param_1.isnull().index
    y = df.param_1[train_index]
    df.drop(['param_1'],axis=1,inplace=True)
    categorical.remove('param_1')
elif target == 'param_2':
    df.set_index('item_id', inplace=True)
    train_index = df.param_2.notnull().index
    test_index = df.param_2.isnull().index
    y = df.param_2[train_index]
    df.drop(['param_2'],axis=1,inplace=True)
    categorical.remove('param_2')
elif target == 'param_3':
    df.set_index('item_id', inplace=True)
    train_index = df.param_3.notnull().index
    test_index = df.param_3.isnull().index
    y = df.param_3[train_index]
    df.drop(['param_3'],axis=1,inplace=True)
    categorical.remove('param_3')

# Predicted Image Top 1
if args.image_top == 'True':
    ##############################################################################################################
    print("Predicted Image Top 1 Feature")
    ##############################################################################################################
    training['image_top_1'] = pd.read_csv("../input/text2image-top-1/train_image_top_1_features.csv", index_col= "item_id")
    testing['image_top_1'] = pd.read_csv("../input/text2image-top-1/test_image_top_1_features.csv", index_col= "item_id")
    df.drop(['image_top_1'], axis=1, inplace=True)
    df = pd.concat([df,pd.concat([training['image_top_1'],testing['image_top_1']],axis=0)], axis=1)

if args.compare == 'True':
    if "pos_title.csv" not in os.listdir("."):
        good_performing_ads = df_train[df_train['deal_probability'] >= 0.90]
        bad_performing_ads = df_train[df_train['deal_probability'] <= 0.05]
        pos_text_cln = list(" ".join(good_performing_ads.title).split(" "))
        neg_text_cln = list(" ".join(bad_performing_ads.title).split(" "))
        tqdm.pandas()
        df['pos_title'] = df['title'].progress_apply(lambda x: sum(np.isin(x.split(" "),pos_text_cln)))
        df['neg_title'] = df['title'].progress_apply(lambda x: sum(np.isin(x.split(" "),neg_text_cln)))
        df['pos_title'].to_csv("pos_title.csv", index=True, header='pos_title')
        df['neg_title'].to_csv("neg_title.csv", index=True, header='neg_title')
    else:
        pos_title = pd.read_csv("pos_title.csv", index_col='item_id').astype(int)
        # neg_title = pd.read_csv("neg_title.csv", index_col='item_id')
        df = pd.concat([df,pos_title], axis=1)
        # df = pd.concat([df,neg_title], axis=1)
        del pos_title
        gc.collect()

if args.deal == 'True':
    ##############################################################################################################
    print("Features Involved with Deal Probability")
    ##############################################################################################################
    # bins of deal probability
    df['avg_deal_by_item_seq_number'] = df['item_seq_number'].map(df_train.groupby(['item_seq_number'])['deal_probability'].transform('mean'))
    df['std_deal_by_item_seq_number'] = df['item_seq_number'].map(df_train.groupby(['item_seq_number'])['deal_probability'].transform('std'))
    df['min_deal_by_item_seq_number'] = df['item_seq_number'].map(df_train.groupby(['item_seq_number'])['deal_probability'].transform('min'))
    df['q1_deal_by_item_seq_number'] = df['item_seq_number'].map(df_train.groupby(['item_seq_number'])['deal_probability'].transform('quantile',q=0.25))
    df['med_deal_by_item_seq_number'] = df['item_seq_number'].map(df_train.groupby(['item_seq_number'])['deal_probability'].transform('quantile',q=0.5))
    df['q3_deal_by_item_seq_number'] = df['item_seq_number'].map(df_train.groupby(['item_seq_number'])['deal_probability'].transform('quantile',q=0.75))
    df['max_deal_by_item_seq_number'] = df['item_seq_number'].map(df_train.groupby(['item_seq_number'])['deal_probability'].transform('max'))
    df['avg_deal_by_item_seq_number'].fillna(-1, inplace=True)
    df['std_deal_by_item_seq_number'].fillna(-1, inplace=True)
    df['min_deal_by_item_seq_number'].fillna(-1, inplace=True)
    df['q1_deal_by_item_seq_number'].fillna(-1, inplace=True)
    df['med_deal_by_item_seq_number'].fillna(-1, inplace=True)
    df['q3_deal_by_item_seq_number'].fillna(-1, inplace=True)
    df['max_deal_by_item_seq_number'].fillna(-1, inplace=True)

    bins = np.linspace(min(df.avg_days_up_user.values)-1,max(df.avg_days_up_user.values),20).astype(float)
    bin_label = ['bin_'+str(i) for i in range(len(bins)-1)]
    df['days_up_bin'] = pd.cut(df.avg_days_up_user,bins, labels=bin_label).astype('category')
    df['avg_deal_by_days_up_bin'] = df['days_up_bin'].map(df.loc[train_index,['days_up_bin','deal_probability']].groupby(['days_up_bin'])['deal_probability'].describe()['mean'])
    df['std_deal_by_days_up_bin'] = df['days_up_bin'].map(df.loc[train_index,['days_up_bin','deal_probability']].groupby(['days_up_bin'])['deal_probability'].describe()['std'])
    df['min_deal_by_days_up_bin'] = df['days_up_bin'].map(df.loc[train_index,['days_up_bin','deal_probability']].groupby(['days_up_bin'])['deal_probability'].describe()['min'])
    df['q1_deal_by_days_up_bin'] = df['days_up_bin'].map(df.loc[train_index,['days_up_bin','deal_probability']].groupby(['days_up_bin'])['deal_probability'].describe()['25%'])
    df['med_deal_by_days_up_bin'] = df['days_up_bin'].map(df.loc[train_index,['days_up_bin','deal_probability']].groupby(['days_up_bin'])['deal_probability'].describe()['50%'])
    df['q3_deal_by_days_up_bin'] = df['days_up_bin'].map(df.loc[train_index,['days_up_bin','deal_probability']].groupby(['days_up_bin'])['deal_probability'].describe()['75%'])
    df['max_deal_by_days_up_bin'] = df['days_up_bin'].map(df.loc[train_index,['days_up_bin','deal_probability']].groupby(['days_up_bin'])['deal_probability'].describe()['max'])
    df['avg_deal_by_days_up_bin'].fillna(-1, inplace=True)
    df['std_deal_by_days_up_bin'].fillna(-1, inplace=True)
    df['min_deal_by_days_up_bin'].fillna(-1, inplace=True)
    df['q1_deal_by_days_up_bin'].fillna(-1, inplace=True)
    df['med_deal_by_days_up_bin'].fillna(-1, inplace=True)
    df['q3_deal_by_days_up_bin'].fillna(-1, inplace=True)
    df['max_deal_by_days_up_bin'].fillna(-1, inplace=True)

    bins = np.linspace(min(df.avg_times_up_user.values),max(df.avg_times_up_user.values),max(df.avg_times_up_user.values)/1.0).astype(float)
    bin_label = ['bin_'+str(i) for i in range(len(bins)-1)]
    df['times_up_bin'] = pd.cut(df.avg_times_up_user,bins, labels=bin_label).astype('category')
    df['avg_deal_by_times_up_bin'] = df['times_up_bin'].map(df.loc[train_index,:].groupby(['times_up_bin'])['deal_probability'].describe()['mean'])
    df['std_deal_by_times_up_bin'] = df['times_up_bin'].map(df.loc[train_index,:].groupby(['times_up_bin'])['deal_probability'].describe()['std'])
    df['min_deal_by_times_up_bin'] = df['times_up_bin'].map(df.loc[train_index,:].groupby(['times_up_bin'])['deal_probability'].describe()['min'])
    df['q1_deal_by_times_up_bin'] = df['times_up_bin'].map(df.loc[train_index,:].groupby(['times_up_bin'])['deal_probability'].describe()['25%'])
    df['med_deal_by_times_up_bin'] = df['times_up_bin'].map(df.loc[train_index,:].groupby(['times_up_bin'])['deal_probability'].describe()['50%'])
    df['q3_deal_by_times_up_bin'] = df['times_up_bin'].map(df.loc[train_index,:].groupby(['times_up_bin'])['deal_probability'].describe()['75%'])
    df['max_deal_by_times_up_bin'] = df['times_up_bin'].map(df.loc[train_index,:].groupby(['times_up_bin'])['deal_probability'].describe()['max'])
    df['avg_deal_by_times_up_bin'].fillna(-1, inplace=True)
    df['std_deal_by_times_up_bin'].fillna(-1, inplace=True)
    df['min_deal_by_times_up_bin'].fillna(-1, inplace=True)
    df['q1_deal_by_times_up_bin'].fillna(-1, inplace=True)
    df['med_deal_by_times_up_bin'].fillna(-1, inplace=True)
    df['q3_deal_by_times_up_bin'].fillna(-1, inplace=True)
    df['max_deal_by_times_up_bin'].fillna(-1, inplace=True)

    temp = df_train.groupby(['parent_category_name','user_type'])['deal_probability'].describe()
    temp.drop('count',axis=1, inplace=True)
    temp.rename(index=str, columns={"mean"  :"avg_deal_by_parent_category_name_user_type", 
                                    "std"   :"std_deal_by_parent_category_name_user_type", 
                                    "min"   :"min_deal_by_parent_category_name_user_type", 
                                    "25%"   :"q1_deal_by_parent_category_name_user_type", 
                                    "50%"   :"med_deal_by_parent_category_name_user_type", 
                                    "75%"   :"q3_deal_by_parent_category_name_user_type", 
                                    "max"   :"max_deal_by_parent_category_name_user_type"},
                            inplace=True)
    df = df.join(temp, on=['parent_category_name','user_type'])
    del temp
    gc.collect()
    df['avg_deal_by_parent_category_name_user_type'].fillna(-1, inplace=True)
    df['std_deal_by_parent_category_name_user_type'].fillna(-1, inplace=True)
    df['min_deal_by_parent_category_name_user_type'].fillna(-1, inplace=True)
    df['q1_deal_by_parent_category_name_user_type'].fillna(-1, inplace=True)
    df['med_deal_by_parent_category_name_user_type'].fillna(-1, inplace=True)
    df['q3_deal_by_parent_category_name_user_type'].fillna(-1, inplace=True)
    df['max_deal_by_parent_category_name_user_type'].fillna(-1, inplace=True)

    temp = df_train.groupby(['region','user_type'])['deal_probability'].describe()
    temp.drop('count',axis=1, inplace=True)
    temp.rename(index=str, columns={"mean"  :"avg_deal_by_region_user_type", 
                                    "std"   :"std_deal_by_region_user_type", 
                                    "min"   :"min_deal_by_region_user_type", 
                                    "25%"   :"q1_deal_by_region_user_type", 
                                    "50%"   :"med_deal_by_region_user_type", 
                                    "75%"   :"q3_deal_by_region_user_type", 
                                    "max"   :"max_deal_by_region_user_type"},
                            inplace=True)
    df = df.join(temp, on=['region','user_type'])
    del temp
    gc.collect()
    df['avg_deal_by_region_user_type'].fillna(-1, inplace=True)
    df['std_deal_by_region_user_type'].fillna(-1, inplace=True)
    df['min_deal_by_region_user_type'].fillna(-1, inplace=True)
    df['q1_deal_by_region_user_type'].fillna(-1, inplace=True)
    df['med_deal_by_region_user_type'].fillna(-1, inplace=True)
    df['q3_deal_by_region_user_type'].fillna(-1, inplace=True)
    df['max_deal_by_region_user_type'].fillna(-1, inplace=True)

    temp = df.loc[train_index,:].groupby(['parent_category_name','times_up_bin'])['deal_probability'].describe()
    temp.drop('count',axis=1, inplace=True)
    temp.rename(index=str, columns={"mean"  :"avg_deal_by_parent_category_name_times_up_bin", 
                                    "std"   :"std_deal_by_parent_category_name_times_up_bin", 
                                    "min"   :"min_deal_by_parent_category_name_times_up_bin", 
                                    "25%"   :"q1_deal_by_parent_category_name_times_up_bin", 
                                    "50%"   :"med_deal_by_parent_category_name_times_up_bin", 
                                    "75%"   :"q3_deal_by_parent_category_name_times_up_bin", 
                                    "max"   :"max_deal_by_parent_category_name_times_up_bin"},
                            inplace=True)
    df = df.join(temp, on=['parent_category_name','times_up_bin'])
    del temp
    gc.collect()
    df['avg_deal_by_parent_category_name_times_up_bin'].fillna(-1, inplace=True)
    df['std_deal_by_parent_category_name_times_up_bin'].fillna(-1, inplace=True)
    df['min_deal_by_parent_category_name_times_up_bin'].fillna(-1, inplace=True)
    df['q1_deal_by_parent_category_name_times_up_bin'].fillna(-1, inplace=True)
    df['med_deal_by_parent_category_name_times_up_bin'].fillna(-1, inplace=True)
    df['q3_deal_by_parent_category_name_times_up_bin'].fillna(-1, inplace=True)
    df['max_deal_by_parent_category_name_times_up_bin'].fillna(-1, inplace=True)

    temp = df.loc[train_index,:].groupby(['parent_category_name','days_up_bin'])['deal_probability'].describe()
    temp.drop('count',axis=1, inplace=True)
    temp.rename(index=str, columns={"mean"  :"avg_deal_by_parent_category_name_days_up_bin", 
                                    "std"   :"std_deal_by_parent_category_name_days_up_bin", 
                                    "min"   :"min_deal_by_parent_category_name_days_up_bin", 
                                    "25%"   :"q1_deal_by_parent_category_name_days_up_bin", 
                                    "50%"   :"med_deal_by_parent_category_name_days_up_bin", 
                                    "75%"   :"q3_deal_by_parent_category_name_days_up_bin", 
                                    "max"   :"max_deal_by_parent_category_name_days_up_bin"},
                            inplace=True)
    df = df.join(temp, on=['parent_category_name','days_up_bin'])
    del temp
    gc.collect()
    df['avg_deal_by_parent_category_name_days_up_bin'].fillna(-1, inplace=True)
    df['std_deal_by_parent_category_name_days_up_bin'].fillna(-1, inplace=True)
    df['min_deal_by_parent_category_name_days_up_bin'].fillna(-1, inplace=True)
    df['q1_deal_by_parent_category_name_days_up_bin'].fillna(-1, inplace=True)
    df['med_deal_by_parent_category_name_days_up_bin'].fillna(-1, inplace=True)
    df['q3_deal_by_parent_category_name_days_up_bin'].fillna(-1, inplace=True)
    df['max_deal_by_parent_category_name_days_up_bin'].fillna(-1, inplace=True)

    df['avg_deal_by_image_top_1'] = df['image_top_1'].map(df_train.groupby(['image_top_1'])['deal_probability'].transform('mean'))
    df['std_deal_by_image_top_1'] = df['image_top_1'].map(df_train.groupby(['image_top_1'])['deal_probability'].transform('std'))
    df['min_deal_by_image_top_1'] = df['image_top_1'].map(df_train.groupby(['image_top_1'])['deal_probability'].transform('min'))
    df['q1_deal_by_image_top_1'] = df['image_top_1'].map(df_train.groupby(['image_top_1'])['deal_probability'].transform('quantile',q=0.25))
    df['med_deal_by_image_top_1'] = df['image_top_1'].map(df_train.groupby(['image_top_1'])['deal_probability'].transform('quantile',q=0.5))
    df['q3_deal_by_image_top_1'] = df['image_top_1'].map(df_train.groupby(['image_top_1'])['deal_probability'].transform('quantile',q=0.75))
    df['max_deal_by_image_top_1'] = df['image_top_1'].map(df_train.groupby(['image_top_1'])['deal_probability'].transform('max'))
    df['avg_deal_by_image_top_1'].fillna(-1, inplace=True)
    df['std_deal_by_image_top_1'].fillna(-1, inplace=True)
    df['min_deal_by_image_top_1'].fillna(-1, inplace=True)
    df['q1_deal_by_image_top_1'].fillna(-1, inplace=True)
    df['med_deal_by_image_top_1'].fillna(-1, inplace=True)
    df['q3_deal_by_image_top_1'].fillna(-1, inplace=True)
    df['max_deal_by_image_top_1'].fillna(-1, inplace=True)

    # bins = np.linspace(min(df.price.values)-1,max(df.price.values),int(max(df.price.values)/0.01)).astype(int)
    bins = np.logspace(min(df.price.values)+1,max(df.price.values),num=100, base=(np.e-1)).astype(float)
    bin_label = ['bin_'+str(i) for i in range(len(bins)-1)]
    df['price_range'] = pd.cut(df.price,bins, labels=bin_label).astype('category')
    df['avg_deal_by_price_range'] = df['price_range'].map(df.loc[train_index,['price_range','deal_probability']].groupby(['price_range'])['deal_probability'].describe()['mean'])
    df['std_deal_by_price_range'] = df['price_range'].map(df.loc[train_index,['price_range','deal_probability']].groupby(['price_range'])['deal_probability'].describe()['std'])
    df['min_deal_by_price_range'] = df['price_range'].map(df.loc[train_index,['price_range','deal_probability']].groupby(['price_range'])['deal_probability'].describe()['min'])
    df['q1_deal_by_price_range'] = df['price_range'].map(df.loc[train_index,['price_range','deal_probability']].groupby(['price_range'])['deal_probability'].describe()['25%'])
    df['med_deal_by_price_range'] = df['price_range'].map(df.loc[train_index,['price_range','deal_probability']].groupby(['price_range'])['deal_probability'].describe()['50%'])
    df['q3_deal_by_price_range'] = df['price_range'].map(df.loc[train_index,['price_range','deal_probability']].groupby(['price_range'])['deal_probability'].describe()['75%'])
    df['max_deal_by_price_range'] = df['price_range'].map(df.loc[train_index,['price_range','deal_probability']].groupby(['price_range'])['deal_probability'].describe()['max'])
    df['avg_deal_by_price_range'].fillna(-1, inplace=True)
    df['std_deal_by_price_range'].fillna(-1, inplace=True)
    df['min_deal_by_price_range'].fillna(-1, inplace=True)
    df['q1_deal_by_price_range'].fillna(-1, inplace=True)
    df['med_deal_by_price_range'].fillna(-1, inplace=True)
    df['q3_deal_by_price_range'].fillna(-1, inplace=True)
    df['max_deal_by_price_range'].fillna(-1, inplace=True)

    bins = np.linspace(min(df.item_seq_number.values)-1,max(df.item_seq_number.values),int(max(df.item_seq_number.values)/500)).astype(int)
    bin_label = ['bin_'+str(i) for i in range(len(bins)-1)]
    df['item_bin'] = pd.cut(df.item_seq_number,bins, labels=bin_label).astype('category')
    df['avg_deal_by_item_seq_number_bin'] = df['item_bin'].map(df.loc[train_index,['item_bin','deal_probability']].groupby(['item_bin'])['deal_probability'].describe()['mean'])
    df['std_deal_by_item_seq_number_bin'] = df['item_bin'].map(df.loc[train_index,['item_bin','deal_probability']].groupby(['item_bin'])['deal_probability'].describe()['std'])
    df['avg_deal_by_item_seq_number_bin'].fillna(-1, inplace=True)
    df['std_deal_by_item_seq_number_bin'].fillna(-1, inplace=True)

    categorical = categorical + ['item_bin','price_range','days_up_bin','times_up_bin']

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
    df = pd.concat([df,pd.concat([training[te_cats],testing[te_cats]],axis=0)], axis=1)

    te_cats = [cat+"_te_price_full" for cat in f_cats]
    target_encode = TargetEncoder(min_samples_leaf=100, smoothing=10, noise_level=0.01,
                                  keep_original=True, cols=f_cats)
    training, testing = target_encode.encode("_te_price_full", training, testing, df['price_full'].iloc[:ntrain])
    df = pd.concat([df,pd.concat([training[te_cats],testing[te_cats]],axis=0)], axis=1)

    te_cats = [cat+"_te_deal" for cat in f_cats]
    target_encode = TargetEncoder(min_samples_leaf=100, smoothing=10, noise_level=0.01,
                                  keep_original=True, cols=f_cats)
    training, testing = target_encode.encode("_te_deal", training, testing, df['deal_probability'].iloc[:ntrain])
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
    del temp
    gc.collect()

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

if args.mean == "True":
    ##############################################################################################################
    print("Statistical Encoding for Categorical Features")
    ############################################################################################################## 
    
    df['avg_avg_times_up_user_by_region'] = df.groupby(['region'])['avg_times_up_user'].transform('mean')
    df['std_avg_times_up_user_by_region'] = df.groupby(['region'])['avg_times_up_user'].transform('std')
    df['var_avg_times_up_user_by_region'] = df.groupby(['region'])['avg_times_up_user'].transform('var')
    df['min_avg_times_up_user_by_region'] = df.groupby(['region'])['avg_times_up_user'].transform('min')
    df['q1_avg_times_up_user_by_region'] = df.groupby(['region'])['avg_times_up_user'].transform('quantile',q=0.25)
    df['med_avg_times_up_user_by_region'] = df.groupby(['region'])['avg_times_up_user'].transform('quantile',q=0.5)
    df['q3_avg_times_up_user_by_region'] = df.groupby(['region'])['avg_times_up_user'].transform('quantile',q=0.75)
    df['max_avg_times_up_user_by_region'] = df.groupby(['region'])['avg_times_up_user'].transform('max')

    df['avg_price_by_avg_times_up_user'] = df.groupby(['avg_times_up_user'])['price'].transform('mean')
    df['std_price_by_avg_times_up_user'] = df.groupby(['avg_times_up_user'])['price'].transform('std')
    df['var_price_by_avg_times_up_user'] = df.groupby(['avg_times_up_user'])['price'].transform('var')
    df['min_price_by_avg_times_up_user'] = df.groupby(['avg_times_up_user'])['price'].transform('min')
    df['q1_price_by_avg_times_up_user'] = df.groupby(['avg_times_up_user'])['price'].transform('quantile',q=0.25)
    df['med_price_by_avg_times_up_user'] = df.groupby(['avg_times_up_user'])['price'].transform('quantile',q=0.5)
    df['q3_price_by_avg_times_up_user'] = df.groupby(['avg_times_up_user'])['price'].transform('quantile',q=0.75)
    df['max_price_by_avg_times_up_user'] = df.groupby(['avg_times_up_user'])['price'].transform('max')
    df['distance_to_avg_price_by_avg_times_up_user'] = df['avg_price_by_avg_times_up_user'] - df['price']

    df['avg_price_by_parent_category_name_avg_times_up_user'] = df.groupby(['parent_category_name','avg_times_up_user'])['price'].transform('mean')
    df['std_price_by_parent_category_name_avg_times_up_user'] = df.groupby(['parent_category_name','avg_times_up_user'])['price'].transform('std')
    df['var_price_by_parent_category_name_avg_times_up_user'] = df.groupby(['parent_category_name','avg_times_up_user'])['price'].transform('var')
    df['min_price_by_parent_category_name_avg_times_up_user'] = df.groupby(['parent_category_name','avg_times_up_user'])['price'].transform('min')
    df['q1_price_by_parent_category_name_avg_times_up_user'] = df.groupby(['parent_category_name','avg_times_up_user'])['price'].transform('quantile',q=0.25)
    df['med_price_by_parent_category_name_avg_times_up_user'] = df.groupby(['parent_category_name','avg_times_up_user'])['price'].transform('quantile',q=0.5)
    df['q3_price_by_parent_category_name_avg_times_up_user'] = df.groupby(['parent_category_name','avg_times_up_user'])['price'].transform('quantile',q=0.75)
    df['max_price_by_parent_category_name_avg_times_up_user'] = df.groupby(['parent_category_name','avg_times_up_user'])['price'].transform('max')
    df['distance_to_avg_price_by_parent_category_name_avg_times_up_user'] = df['avg_price_by_parent_category_name_avg_times_up_user'] - df['price']

    df['avg_price_by_avg_days_up_user'] = df.groupby(['avg_days_up_user'])['price'].transform('mean')
    df['std_price_by_avg_days_up_user'] = df.groupby(['avg_days_up_user'])['price'].transform('std')
    df['var_price_by_avg_days_up_user'] = df.groupby(['avg_days_up_user'])['price'].transform('var')
    df['min_price_by_avg_days_up_user'] = df.groupby(['avg_days_up_user'])['price'].transform('min')
    df['q1_price_by_avg_days_up_user'] = df.groupby(['avg_days_up_user'])['price'].transform('quantile',q=0.25)
    df['med_price_by_avg_days_up_user'] = df.groupby(['avg_days_up_user'])['price'].transform('quantile',q=0.5)
    df['q3_price_by_avg_days_up_user'] = df.groupby(['avg_days_up_user'])['price'].transform('quantile',q=0.75)
    df['max_price_by_avg_days_up_user'] = df.groupby(['avg_days_up_user'])['price'].transform('max')
    df['distance_to_avg_price_by_avg_days_up_user'] = df['avg_price_by_avg_days_up_user'] - df['price']
    
    df['avg_price_by_parent_category_name_avg_days_up_user'] = df.groupby(['parent_category_name','avg_days_up_user'])['price'].transform('mean')
    df['std_price_by_parent_category_name_avg_days_up_user'] = df.groupby(['parent_category_name','avg_days_up_user'])['price'].transform('std')
    df['var_price_by_parent_category_name_avg_days_up_user'] = df.groupby(['parent_category_name','avg_days_up_user'])['price'].transform('var')
    df['min_price_by_parent_category_name_avg_days_up_user'] = df.groupby(['parent_category_name','avg_days_up_user'])['price'].transform('min')
    df['q1_price_by_parent_category_name_avg_days_up_user'] = df.groupby(['parent_category_name','avg_days_up_user'])['price'].transform('quantile',q=0.25)
    df['med_price_by_parent_category_name_avg_days_up_user'] = df.groupby(['parent_category_name','avg_days_up_user'])['price'].transform('quantile',q=0.5)
    df['q3_price_by_parent_category_name_avg_days_up_user'] = df.groupby(['parent_category_name','avg_days_up_user'])['price'].transform('quantile',q=0.75)
    df['max_price_by_parent_category_name_avg_days_up_user'] = df.groupby(['parent_category_name','avg_days_up_user'])['price'].transform('max')
    df['distance_to_avg_price_by_parent_category_name_avg_days_up_user'] = df['avg_price_by_parent_category_name_avg_days_up_user'] - df['price']

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

    df['avg_price_by_city_image_top_1'] = df.groupby(['city','image_top_1'])['price'].transform('mean')
    df['std_price_by_city_image_top_1'] = df.groupby(['city','image_top_1'])['price'].transform('std')
    df['var_price_by_city_image_top_1'] = df.groupby(['city','image_top_1'])['price'].transform('var')
    df['min_price_by_city_image_top_1'] = df.groupby(['city','image_top_1'])['price'].transform('min')
    df['q1_price_by_city_image_top_1'] = df.groupby(['city','image_top_1'])['price'].transform('quantile',q=0.25)
    df['med_price_by_city_image_top_1'] = df.groupby(['city','image_top_1'])['price'].transform('quantile',q=0.5)
    df['q3_price_by_city_image_top_1'] = df.groupby(['city','image_top_1'])['price'].transform('quantile',q=0.75)
    df['max_price_by_city_image_top_1'] = df.groupby(['city','image_top_1'])['price'].transform('max')
    df['distance_to_avg_price_by_city_image_top_1'] = df['avg_price_by_city_image_top_1'] - df['price']

    df['avg_price_by_city_image_top_1_day_of_week'] = df.groupby(['city','image_top_1','day_of_week'])['price'].transform('mean')
    df['std_price_by_city_image_top_1_day_of_week'] = df.groupby(['city','image_top_1','day_of_week'])['price'].transform('std')
    df['var_price_by_city_image_top_1_day_of_week'] = df.groupby(['city','image_top_1','day_of_week'])['price'].transform('var')
    df['min_price_by_city_image_top_1_day_of_week'] = df.groupby(['city','image_top_1','day_of_week'])['price'].transform('min')
    df['q1_price_by_city_image_top_1_day_of_week'] = df.groupby(['city','image_top_1','day_of_week'])['price'].transform('quantile',q=0.25)
    df['med_price_by_city_image_top_1_day_of_week'] = df.groupby(['city','image_top_1','day_of_week'])['price'].transform('quantile',q=0.5)
    df['q3_price_by_city_image_top_1_day_of_week'] = df.groupby(['city','image_top_1','day_of_week'])['price'].transform('quantile',q=0.75)
    df['max_price_by_city_image_top_1_day_of_week'] = df.groupby(['city','image_top_1','day_of_week'])['price'].transform('max')
    df['distance_to_avg_price_by_city_image_top_1_day_of_week'] = df['avg_price_by_city_image_top_1_day_of_week'] - df['price']

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

    df['avg_image_top_1_by_city'] = df.groupby(['city'])['image_top_1'].transform('mean')
    df['std_image_top_1_by_city'] = df.groupby(['city'])['image_top_1'].transform('std')
    df['var_image_top_1_by_city'] = df.groupby(['city'])['image_top_1'].transform('var')
    df['min_image_top_1_by_city'] = df.groupby(['city'])['image_top_1'].transform('min')
    df['q1_image_top_1_by_city'] = df.groupby(['city'])['image_top_1'].transform('quantile',q=0.25)
    df['med_image_top_1_by_city'] = df.groupby(['city'])['image_top_1'].transform('quantile',q=0.5)
    df['q3_image_top_1_by_city'] = df.groupby(['city'])['image_top_1'].transform('quantile',q=0.75)
    df['max_image_top_1_by_city'] = df.groupby(['city'])['image_top_1'].transform('max')
    df['distance_to_avg_image_top_1_by_city'] = df['avg_image_top_1_by_city'] - df['image_top_1']

    df['avg_image_top_1_by_city_user_type'] = df.groupby(['city','user_type'])['image_top_1'].transform('mean')
    df['std_image_top_1_by_city_user_type'] = df.groupby(['city','user_type'])['image_top_1'].transform('std')
    df['var_image_top_1_by_city_user_type'] = df.groupby(['city','user_type'])['image_top_1'].transform('var')
    df['min_image_top_1_by_city_user_type'] = df.groupby(['city','user_type'])['image_top_1'].transform('min')
    df['q1_image_top_1_by_city_user_type'] = df.groupby(['city','user_type'])['image_top_1'].transform('quantile',q=0.25)
    df['med_image_top_1_by_city_user_type'] = df.groupby(['city','user_type'])['image_top_1'].transform('quantile',q=0.5)
    df['q3_image_top_1_by_city_user_type'] = df.groupby(['city','user_type'])['image_top_1'].transform('quantile',q=0.75)
    df['max_image_top_1_by_city_user_type'] = df.groupby(['city','user_type'])['image_top_1'].transform('max')
    df['distance_to_avg_image_top_1_by_city'] = df['avg_image_top_1_by_city'] - df['image_top_1']

if args.wordbatch == 'True':
    ##############################################################################################################
    print("WordBatch Features")
    ##############################################################################################################
    from sklearn.metrics import mean_squared_error
    from math import sqrt

    kf = KFold(ntrain, n_folds=NFOLDS, shuffle=True, random_state=SEED)
    russian_stopwords = {x: 1 for x in stopwords.words('russian')}
    def normalize_text(text):
        text = text.lower().strip()
        for s in string.punctuation:
            text = text.replace(s, ' ')
        text = text.strip().split(' ')
        return u' '.join(x for x in text if len(x) > 1 and x not in russian_stopwords)

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

    def ridgeSolver(X_train, X_test, y_train, solver_alg):
        oof_train = np.zeros((ntrain,))
        oof_test = np.zeros((ntest,))
        oof_test_skf = np.empty((NFOLDS, ntest))

        for i, (train_ind, test_ind) in enumerate(kf):
            print('Ridge Regression, Fold {}'.format(i))
            x_tr = X_train[train_ind]
            y_tr = y_train[train_ind]
            x_te = X_train[test_ind]
            if solver_alg == "sag":
                intercept = True
            else:
                intercept = False
            model = Ridge(solver=solver_alg, fit_intercept=intercept, random_state=205, alpha=3.3, copy_X=True)
            model.fit(x_tr, y_tr)
            oof_train[test_ind] = model.predict(x_te)
            oof_test_skf[i, :] = model.predict(X_test)

        oof_test[:] = oof_test_skf.mean(axis=0)
        oof_train = oof_train.reshape(-1, 1)
        oof_test = oof_test.reshape(-1, 1)
        rms = sqrt(mean_squared_error(y, oof_train))
        print('Ridge OOF {}, RMSE: {}'.format(solver_alg, rms))
        return np.concatenate([oof_train, oof_test])

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
        stemmed_description = pd.read_csv("stemmed_description.csv", index_col='item_id')
        stemmed_title = pd.read_csv("stemmed_title.csv", index_col='item_id')
        df = pd.concat([df,stemmed_description], axis=1)
        df = pd.concat([df,stemmed_title], axis=1)

    # wb = wordbatch.WordBatch(normalize_text, extractor=(WordBag, {"hash_ngrams": 2,
    #                                                               "hash_ngrams_weights": [1.5, 1.0],
    #                                                               "hash_size": 2 ** 29,
    #                                                               "norm": None,
    #                                                               "tf": 'binary',
    #                                                               "idf": None,
    #                                                               }), procs=8)
    # wb.dictionary_freeze = True
    # X_title = wb.fit_transform(df['title'].fillna(''))
    # del(wb)
    # gc.collect()
    # mask = np.where(X_title.getnnz(axis=0) > 3)[0]
    # X_title = X_title[:, mask]
    # print(X_title.shape)

    if "title_ridge_preds_sag.csv" not in os.listdir("."):
        df['title_ridge_preds_sag'] = ridgeSolver(X_title[:ntrain], X_title[ntrain:], y, "sag")
        df['title_ridge_preds_sag'].to_csv("title_ridge_preds_sag.csv", index=True, header='title_ridge_preds_sag')
        gc.collect()
    else:
        title_ridge_preds_sag = pd.read_csv("title_ridge_preds_sag.csv", index_col='item_id')
        df = pd.concat([df,title_ridge_preds_sag], axis=1)

    if "title_ridge_preds_saga.csv" not in os.listdir("."):
        df['title_ridge_preds_saga'] = ridgeSolver(X_title[:ntrain], X_title[ntrain:], y, "saga")
        df['title_ridge_preds_saga'].to_csv("title_ridge_preds_saga.csv", index=True, header='title_ridge_preds_saga')
        gc.collect()
    else:
        title_ridge_preds_saga = pd.read_csv("title_ridge_preds_saga.csv", index_col='item_id')
        df = pd.concat([df,title_ridge_preds_saga], axis=1)

    if "title_ridge_preds_lsqr.csv" not in os.listdir("."):
        df['title_ridge_preds_lsqr'] = ridgeSolver(X_title[:ntrain], X_title[ntrain:], y, "lsqr")
        df['title_ridge_preds_lsqr'].to_csv("title_ridge_preds_lsqr.csv", index=True, header='title_ridge_preds_lsqr')
        gc.collect()
    else:
        title_ridge_preds_lsqr = pd.read_csv("title_ridge_preds_lsqr.csv", index_col='item_id')
        df = pd.concat([df,title_ridge_preds_lsqr], axis=1)

    if "title_ridge_preds_sparse_cg.csv" not in os.listdir("."):
        df['title_ridge_preds_sparse_cg'] = ridgeSolver(X_title[:ntrain], X_title[ntrain:], y, "sparse_cg")
        df['title_ridge_preds_sparse_cg'].to_csv("title_ridge_preds_sparse_cg.csv", index=True, header='title_ridge_preds_sparse_cg')
        gc.collect()
    else:
        title_ridge_preds_sparse_cg = pd.read_csv("title_ridge_preds_sparse_cg.csv", index_col='item_id')
        df = pd.concat([df,title_ridge_preds_sparse_cg], axis=1)

    # wb = wordbatch.WordBatch(normalize_text, extractor=(WordBag, {"hash_ngrams": 2,
    #                                                               "hash_ngrams_weights": [1.0, 1.0],
    #                                                               "hash_size": 2 ** 28,
    #                                                               "norm": "l2",
    #                                                               "tf": 1.0,
    #                                                               "idf": None
    #                                                               }), procs=8)
    # wb.dictionary_freeze = True
    # X_description = wb.fit_transform(df['description'].fillna(''))
    # del(wb)
    # gc.collect()
    # mask = np.where(X_description.getnnz(axis=0) > 8)[0]
    # X_description = X_description[:, mask]
    # print(X_description.shape)

    if "description_ridge_preds_sag.csv" not in os.listdir("."):
        df['description_ridge_preds_sag'] = ridgeSolver(X_title[:ntrain], X_title[ntrain:], y, "sag")
        df['description_ridge_preds_sag'].to_csv("description_ridge_preds_sag.csv", index=True, header='description_ridge_preds_sag')
        gc.collect()
    else:
        title_ridge_preds_sag = pd.read_csv("description_ridge_preds_sag.csv", index_col='item_id')
        df = pd.concat([df,title_ridge_preds_sag], axis=1)

    if "description_ridge_preds_saga.csv" not in os.listdir("."):
        df['description_ridge_preds_saga'] = ridgeSolver(X_title[:ntrain], X_title[ntrain:], y, "saga")
        df['description_ridge_preds_saga'].to_csv("description_ridge_preds_saga.csv", index=True, header='description_ridge_preds_saga')
        gc.collect()
    else:
        description_ridge_preds_saga = pd.read_csv("description_ridge_preds_saga.csv", index_col='item_id')
        df = pd.concat([df,description_ridge_preds_saga], axis=1)

    if "description_ridge_preds_lsqr.csv" not in os.listdir("."):
        df['description_ridge_preds_lsqr'] = ridgeSolver(X_title[:ntrain], X_title[ntrain:], y, "lsqr")
        df['description_ridge_preds_lsqr'].to_csv("description_ridge_preds_lsqr.csv", index=True, header='description_ridge_preds_lsqr')
        gc.collect()
    else:
        description_ridge_preds_lsqr = pd.read_csv("description_ridge_preds_lsqr.csv", index_col='item_id')
        df = pd.concat([df,description_ridge_preds_lsqr], axis=1)

    if "description_ridge_preds_sparse_cg.csv" not in os.listdir("."):
        df['description_ridge_preds_sparse_cg'] = ridgeSolver(X_title[:ntrain], X_title[ntrain:], y, "sparse_cg")
        df['description_ridge_preds_sparse_cg'].to_csv("description_ridge_preds_sparse_cg.csv", index=True, header='description_ridge_preds_sparse_cg')
        gc.collect()
    else:
        description_ridge_preds_sparse_cg = pd.read_csv("description_ridge_preds_sparse_cg.csv", index_col='item_id')
        df = pd.concat([df,description_ridge_preds_sparse_cg], axis=1)        

if args.tfidf == "True":
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

if args.sparse == "True":
    temp_train = df.loc[train_index,:]
    temp_test = df.loc[test_index,:]
    tfvocab = df.columns.tolist() + tfvocab
    del df
    X = hstack([csr_matrix(temp_train.values),ready_df[0:train_index.shape[0]]]) # Sparse Matrix
    testing = hstack([csr_matrix(temp_test.values),ready_df[train_index.shape[0]:]])
    del ready_df
    gc.collect();
    feat = pd.read_csv('feature.csv', index_col='Unnamed: 0')
    index_list = list(feat.index)
    tfvocab = list(feat.feature.values)
    X = X.tocsr()[:,index_list]
    testing = testing.tocsr()[:,index_list]
else:
    gc.collect();
    tfvocab = df.columns.tolist()
    X = df.loc[train_index,:]
    testing = df.loc[test_index,:]
    del df #Make room for more memory
    X = X.values 
    testing = testing.values
del training
gc.collect();


if args.build_features == "True":
    sys.exit(1)

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
    'objective': objective,
    # 'metric': 'rmse',
    # 'metric': 'binary_logloss',
    'metric': metric,
    # 'max_depth': 15,
    'num_leaves':500,
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
        num_boost_round=20000,
        valid_sets=[lgbtrain, lgbvalid],
        valid_names=['train','valid'],
        learning_rates=lambda iter:0.1 * (0.999 ** iter),
        early_stopping_rounds=50,
        verbose_eval=100
    )

    model.save_model('model_{}.txt'.format(i));i += 1
    prediction = model.predict(X[valid])
    validation_score = np.sqrt(metrics.mean_squared_error(y[valid], model.predict(X[valid])))
    print('Fold {}, RMSE: {}'.format(i,validation_score))
    cv_score += validation_score
    models.append(model)
    feature = pd.DataFrame(data={'feature':model.feature_name(),'importance':model.feature_importance()})
        # print(feature.sort_values('importance'))
    if args.test == "True":
        sys.exit(1)

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
