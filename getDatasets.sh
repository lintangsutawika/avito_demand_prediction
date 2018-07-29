mkdir input
mkdir input/avito-demand-prediction
kaggle competitions download -c avito-demand-prediction -f train.csv -p input/avito-demand-prediction
kaggle competitions download -c avito-demand-prediction -f test.csv -p input/avito-demand-prediction

mkdir input/image-confidence
kaggle kernels output -k sukhyun9673/export-image-features -p input/image-confidence

kaggle kernels output -k sukhyun9673/extracting-image-features-test -p input/image-confidence

mkdir input/aggregated
kaggle kernels output -k bminixhofer/aggregated-features-lightgbm -p input/aggregated
rm input/aggregated/submission.csv

mkdir input/text2image-top-1
kaggle kernels output -k christofhenkel/text2image-top-1 -p input/text2image-top-1
rm input/text2image-top-1/embedding_dict.p input/text2image-top-1/model.hdf5

mkdir input/image-features/vgg-test
kaggle kernels output -k bguberfain/vgg16-test-features -p input/image-features/vgg-test

mkdir input/image-features/vgg-train
kaggle kernels output -k bguberfain/vgg16-train-features -p input/image-features/vgg-train