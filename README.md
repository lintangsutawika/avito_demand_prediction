# Avito Demand Prediction Challenge

Link: https://www.kaggle.com/c/avito-demand-prediction
Bronze submission (Blended with other high score low correlation submission)

# Downloading Datasets

It's advise that you have access to the competition materials (by accepting terms and aggreements in the competition page) and installed Kaggle's API. 
To download, simply run `getDatasets.sh`. You might need to use `sudo chmod 777 getDatasets.sh` first.
```
./getDatasets.sh
```
or

```
sh getDatasets.sh
```

You will notice that `image_confidence_train.csv` and `image_confidence_test.csv` is unavailable, this is is a custom dataset obtained by inferences of top ImageNet classifiers like ResNets and such. I highly recommend you tweek this one yourself.

# Running Model Training

The model builds upon various features with some generously provided by the amazing members of the Kaggle community.
Each feature category can be turned on or off depending on your experiments. Default option for all is `False`
```
python model.py --build_features True --image_top True
```

`build_features` : Only builds features and stops program, useful for exprimenting on a Jupyter notebook or with iPython.
`image_top` : Predicts missing image-top-1 features from text features. Adapted from https://www.kaggle.com/christofhenkel/text2image-top-1.
`agg_feat` : Engineered features from extending datasets in the competition. Adapted from Adapted from https://www.kaggle.com/bminixhofer/aggregated-features-lightgbm.
`text` : Text features including characters, emoji, and punctuations.
`categorical` : Encode categorical features.
`cat2vec` : Embed categorical features.
`mean` : Statistical features that center around average of certain features.
`target` : Encode categorical features based on their target value.
`wordbatch` : Language features.
`image` : Image related features.
`sparse` : Include sparse features.
`deal` : Engineered features based on deal probability of entry.
`compare` : Compare titles with words found in good probability entries and in bad probability entries.
`tfidf` : Use TFIDF Features.
`test` : Stops program upon completing training. Does not output a submission.
`binary` : Transforms target value to binary.
`xentropy` : Training objective is cross entropy instead of the default regression.
`vgg` : Use features obtained from pretrained VGG network. Adapted from https://www.kaggle.com/bguberfain/vgg16-train-features/code and https://www.kaggle.com/bguberfain/vgg16-test-features/code.
