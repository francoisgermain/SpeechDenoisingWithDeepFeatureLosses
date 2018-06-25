#!/bin/bash

## DOWNLOAD THE DATASETS
mkdir -p dataset_zip
pushd dataset_zip
# VALIDATION DATASET
if [ ! -f clean_testset_wav.zip ]; then
    wget https://datashare.is.ed.ac.uk/bitstream/handle/10283/2791/clean_testset_wav.zip
fi
if [ ! -f noisy_testset_wav.zip ]; then
    wget https://datashare.is.ed.ac.uk/bitstream/handle/10283/2791/noisy_testset_wav.zip
fi
popd

## INFLATE DATA
mkdir -p dataset_tmp
pushd dataset_tmp
unzip -q -j ../dataset_zip/clean_testset_wav.zip -d valset_clean
unzip -q -j ../dataset_zip/noisy_testset_wav.zip -d valset_noisy
popd

## RESAMPLE
declare -a arr=("valset_clean" "valset_noisy")

mkdir -p dataset
pushd dataset_tmp
for d in */; do
    mkdir -p "../dataset/$d"
    pushd "$d"
    for f in *.wav; do
        sox "$f" -e float -b 32 "../../dataset/$d$f" rate -v -I 16000
    done
    popd
done
popd

# REMOVE TMP DATA
rm -r dataset_tmp
