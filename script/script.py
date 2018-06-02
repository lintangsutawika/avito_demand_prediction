#Initially forked from Bojan's kernel here: https://www.kaggle.com/tunguz/bow-meta-text-and-dense-features-lb-0-2242/code
#improvement using kernel from Nick Brook's kernel here: https://www.kaggle.com/nicapotato/bow-meta-text-and-dense-features-lgbm
#Used oof method from Faron's kernel here: https://www.kaggle.com/mmueller/stacking-starter?scriptVersionId=390867
#Used some text cleaning method from Muhammad Alfiansyah's kernel here: https://www.kaggle.com/muhammadalfiansyah/push-the-lgbm-v19
import time
notebookstart= time.time()

from tqdm import tqdm
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc
print("Data:\n",os.listdir("../input"))

# Models Packages
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn import feature_selection
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# Gradient Boosting
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.cross_validation import KFold
from sklearn.model_selection import KFold as KFOLD

# Tf-Idf
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from scipy.sparse import hstack, csr_matrix
from nltk.corpus import stopwords

# Viz
import seaborn as sns
import matplotlib.pyplot as plt
import re
import string

NFOLDS = 5
SEED = 42
VALID = False

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
        print('\nFold {}'.format(i))
        x_tr = x_train[train_index]
        y_tr = y[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

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

print("\nData Load Stage")
training = pd.read_csv('../input/avito-demand-prediction/train.csv', index_col = "item_id", parse_dates = ["activation_date"])
training['image_top_1'] = pd.read_csv("../input/text2image-top-1/train_image_top_1_features.csv", index_col= "item_id")
traindex = training.index

testing = pd.read_csv('../input/avito-demand-prediction/test.csv', index_col = "item_id", parse_dates = ["activation_date"])
testing['image_top_1'] = pd.read_csv("../input/text2image-top-1/test_image_top_1_features.csv", index_col= "item_id")
testdex = testing.index

ntrain = training.shape[0]
ntest = testing.shape[0]

kf = KFold(ntrain, n_folds=NFOLDS, shuffle=True, random_state=SEED)

y = training['deal_probability']
testing['deal_probability'] = -1
# training.drop("deal_probability",axis=1, inplace=True)
print('Train shape: {} Rows, {} Columns'.format(*training.shape))
print('Test shape: {} Rows, {} Columns'.format(*testing.shape))

print("Combine Train and Test")
df = pd.concat([training,testing],axis=0)
del training, testing
gc.collect()
print('\nAll Data shape: {} Rows, {} Columns'.format(*df.shape))

##############################################################################################################
print("Feature Engineering")
##############################################################################################################
df["price"] = np.log(df["price"]+0.001)
df["price"].fillna(df.price.mean(),inplace=True)
df["image_top_1"].fillna(-999,inplace=True)

##############################################################################################################
print("\nCreate Time Variables")
##############################################################################################################
df["week_of_year"] = df['activation_date'].dt.week
df["day_of_month"] = df['activation_date'].dt.day
df["day_of_week"] = df['activation_date'].dt.weekday

df.drop(["activation_date","image"],axis=1,inplace=True)

##############################################################################################################
print("\nEncode Variables")
##############################################################################################################
categorical = ["user_id","region","city","parent_category_name","category_name","user_type","image_top_1","param_1","param_2","param_3"]
print("Encoding :",categorical)

# Encoder:
lbl = preprocessing.LabelEncoder()
for col in categorical:
    df[col].fillna('Unknown')
    df[col] = lbl.fit_transform(df[col].astype(str))

print("\nText Features")

# Feature Engineering

##############################################################################################################
# Meta Text Features
##############################################################################################################
textfeats = ["description", "title"]
df['desc_punc'] = df['description'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))

df['title'] = df['title'].apply(lambda x: cleanName(x))
df["description"]   = df["description"].apply(lambda x: cleanName(x))

df['desc_nchar'] = df["description"].apply(lambda x: len(str(x)))
df['desc_nwords'] = df["description"].apply(lambda x: len(x.split()))

agg_cols = ['region', 'city', 'parent_category_name',
            'category_name','image_top_1', 'user_type',
            'item_seq_number','day_of_month','day_of_week','week_of_year']

for category in tqdm(agg_cols):
    gp = df.loc[traindex,:].groupby(category)['deal_probability']
    mean = gp.mean()
    std  = gp.std()
    df[category + '_deal_probability_avg'] = df[category].map(mean)
    df[category + '_deal_probability_std'] = df[category].map(std)

for category in tqdm(agg_cols):
    gp = df.loc[traindex,:].groupby(category)['price']
    mean = gp.mean()
    df[category + '_price_avg'] = df[category].map(mean)

df.drop("deal_probability",axis=1, inplace=True)

for cols in textfeats:
    df[cols] = df[cols].astype(str)
    df[cols] = df[cols].astype(str).fillna('missing') # FILL NA
    df[cols] = df[cols].str.lower() # Lowercase all text, so that capitalized words dont get treated differently
    df[cols + '_num_words'] = df[cols].apply(lambda comment: len(comment.split())) # Count number of Words
    df[cols + '_num_unique_words'] = df[cols].apply(lambda comment: len(set(w for w in comment.split())))
    df[cols + '_words_vs_unique'] = df[cols+'_num_unique_words'] / df[cols+'_num_words'] * 100 # Count Unique Words

##############################################################################################################
print("\n[TF-IDF] Term Frequency Inverse Document Frequency Stage")
##############################################################################################################
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
tfvocab = vectorizer.get_feature_names()
print("Vectorization Runtime: %0.2f Minutes"%((time.time() - start_vect)/60))

# Drop Text Cols
textfeats = ["description", "title"]
df.drop(textfeats, axis=1,inplace=True)

from sklearn.metrics import mean_squared_error
from math import sqrt

ridge_params = {'alpha':20.0, 'fit_intercept':True, 'normalize':False, 'copy_X':True,
                'max_iter':None, 'tol':0.001, 'solver':'auto', 'random_state':SEED}

#Ridge oof method from Faron's kernel
#I was using this to analyze my vectorization, but figured it would be interesting to add the results back into the dataset
#It doesn't really add much to the score, but it does help lightgbm converge faster
ridge = SklearnWrapper(clf=Ridge, seed = SEED, params = ridge_params)
ridge_oof_train, ridge_oof_test = get_oof(ridge, ready_df[:ntrain], y, ready_df[ntrain:])

rms = sqrt(mean_squared_error(y, ridge_oof_train))
print('Ridge OOF RMSE: {}'.format(rms))

print("Modeling Stage")

ridge_preds = np.concatenate([ridge_oof_train, ridge_oof_test])
df['ridge_preds'] = ridge_preds

# Combine Dense Features with Sparse Text Bag of Words Features
X = hstack([csr_matrix(df.loc[traindex,:].values),ready_df[0:traindex.shape[0]]]) # Sparse Matrix
testing = hstack([csr_matrix(df.loc[testdex,:].values),ready_df[traindex.shape[0]:]])
tfvocab = df.columns.tolist() + tfvocab
for shape in [X,testing]:
    print("{} Rows and {} Cols".format(*shape.shape))
print("Feature Names Length: ",len(tfvocab))
del df,ridge_preds,vectorizer,ready_df
gc.collect();

############################################################################
print("\nModeling Stage")
############################################################################
print("Light Gradient Boosting Regressor")
n_rounds = 15000
lgbm_params =  {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'max_depth': 15,
    'num_leaves': 33,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.8,
    # 'bagging_freq': 5,
    'learning_rate': 0.019,
    'verbose': 0
}

cv_score = 0
temp_prediction = []
models = []
kf_ = KFOLD(n_splits=NFOLDS, shuffle=True, random_state=SEED)
i = 0
for train, valid in kf_.split(X):
    # LGBM Dataset Formatting
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
        num_boost_round=n_rounds,
        valid_sets=[lgbtrain, lgbvalid],
        valid_names=['train','valid'],
        early_stopping_rounds=200,
        verbose_eval=200
    )

    model.save_model('model_{}.txt'.format(i));i += 1
    validation_score = np.sqrt(metrics.mean_squared_error(y[valid], model.predict(X[valid])))
    print('RMSE: {}'.format(validation_score))
    cv_score = cv_score + validation_score
    models.append(model)
############################################################################

# Feature Importance Plot
f, ax = plt.subplots(figsize=[7,10])
lgb.plot_importance(models[0], max_num_features=50, ax=ax)
plt.title("Light GBM Feature Importance")
plt.savefig('feature_import.png')

print("cv_score: {}".format(cv_score/NFOLDS))
# make prediction
predictions = 0
# model = lgb.Booster(model_file='model.txt')
for model in models:
    predictions = predictions + np.asarray(model.predict(testing))

pred_test = predictions/NFOLDS
lgb_predict = pd.DataFrame({"item_id":testdex})
lgb_predict["deal_probability"] = pred_test
lgb_predict['deal_probability'].clip(0.0, 1.0, inplace=True) # Between 0 and 1

#Mixing lightgbm with ridge. I haven't really tested if this improves the score or not
#blend = 0.95*lgpred + 0.05*ridge_oof_test[:,0]
lgb_predict.to_csv("submission.csv", index=False)
print("Notebook Runtime: %0.2f Minutes"%((time.time() - notebookstart)/60))
