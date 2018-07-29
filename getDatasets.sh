mkdir input/image-confidence
kaggle kernels output -k sukhyun9673/export-image-features -p .
mv *_blurrness.csv input/image-confidence

kaggle kernels output -k sukhyun9673/extracting-image-features-test -p .
mv Image_KP_SCORES.csv input/image-confidence

mkdir input/aggregated
kaggle kernels output -k bminixhofer/aggregated-features-lightgbm -p .
rm submission.csv
mv aggregated-features.csv input/aggregated

mkdir input/text2image-top-1
kaggle kernels output -k christofhenkel/text2image-top-1 -p .
rm embedding_dict.p model.hdf5
mv *_image_top_1_features.csv input/text2image-top-1

mkdir input/image-features/vgg-test
kaggle kernels output -k bguberfain/vgg16-test-features -p .
mv features.npz input/image-features/vgg-test

mkdir input/image-features/vgg-train
kaggle kernels output -k bguberfain/vgg16-train-features -p .
mv features.npz input/image-features/vgg-train