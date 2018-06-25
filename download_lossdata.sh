#!/bin/bash

## DOWNLOAD THE DATASETS
mkdir -p dataset_zip
pushd dataset_zip

# TRAINING ASC DATASET
for i in `seq 1 8`; 
do
    if [ ! -f TUT-acoustic-scenes-2016-development.audio.$i.zip ]; then
        wget https://zenodo.org/record/45739/files/TUT-acoustic-scenes-2016-development.audio.$i.zip
    fi
done 
if [ ! -f TUT-acoustic-scenes-2016-development.meta.zip ]; then
    wget https://zenodo.org/record/45739/files/TUT-acoustic-scenes-2016-development.meta.zip
fi

# TESTING ASC DATASET
for i in `seq 1 3`; 
do
    if [ ! -f TUT-acoustic-scenes-2016-evaluation.audio.$i.zip ]; then
        wget https://zenodo.org/record/165995/files/TUT-acoustic-scenes-2016-evaluation.audio.$i.zip
    fi
done 
if [ ! -f TUT-acoustic-scenes-2016-evaluation.meta.zip ]; then
    wget https://zenodo.org/record/165995/files/TUT-acoustic-scenes-2016-evaluation.meta.zip
fi

# DAT DATASET
if [ ! -f chime_home.tar.gz ]; then
    wget https://archive.org/download/chime-home/chime_home.tar.gz
fi

popd

## INFLATE DATA

# ASC DATA
mkdir -p asc_tmp
pushd asc_tmp
# ASC TRAINING
for i in `seq 1 8`;
do
    unzip -q -j ../dataset_zip/TUT-acoustic-scenes-2016-development.audio.$i.zip -d trainset
done
unzip -q -p ../dataset_zip/TUT-acoustic-scenes-2016-development.meta.zip TUT-acoustic-scenes-2016-development/meta.txt > trainset/meta.txt

# ASC TESTING
for i in `seq 1 3`;
do
    unzip -q -j ../dataset_zip/TUT-acoustic-scenes-2016-evaluation.audio.$i.zip -d valset
done
unzip -q -p ../dataset_zip/TUT-acoustic-scenes-2016-evaluation.meta.zip TUT-acoustic-scenes-2016-evaluation/meta.txt > valset/meta.txt

popd

# DAT DATA
tar -xzf dataset_zip/chime_home.tar.gz -C ./

## RESAMPLE
mkdir -p dataset/asc
pushd asc_tmp
for d in */; do
    mkdir -p "../dataset/asc/$d"
    pushd "$d"
    for f in *.wav; do
        sox "$f" -e float -b 32 "../../dataset/asc/$d$f" rate -v -I 16000
    done
    cp "meta.txt" "../../dataset/asc/$dmeta.txt"
    popd
done
popd

mkdir -p dataset/dat
pushd chime_home
for d in */; do
    pushd "$d"
    for f in *.16kHz.wav; do
        ff=$(echo $f | cut -d '.' -f 1-2,4) 
        sox "$f" -e float -b 32 "../../dataset/dat/$ff" #ALREADY AT 16KHZ
    done
    popd
done
cp "development_chunks_refined.csv" "../dataset/dat/development_chunks_refined.csv"
cp "evaluation_chunks_refined.csv" "../dataset/dat/evaluation_chunks_refined.csv"
popd

# REMOVE TMP DATA
rm -r asc_tmp
rm -r chime_home


