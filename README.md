<a name="top"></a>
#  Speech Denoising with Deep Feature Losses ([arXiv](https://arxiv.org/abs/1806.10522), [sound examples](https://ccrma.stanford.edu/~francois/SpeechDenoisingWithDeepFeatureLosses/))
This is a Tensorflow implementation of our [Speech Denoising Convolutional Neural Network trained with Deep Feature Losses](https://arxiv.org/abs/1806.10522).

Contact: [François Germain](mailto:francois@ccrma.stanford.edu)

## Table of contents
1. [Citation](#citation)
2. [Setup](#setup)
3. [Denoising scripts](#scripts)
4. [Models](#models)
5. [Noisy data](#data)
6. [Deep feature loss](#feature-loss)
7. [Notes](#notes)
8. [SoX installation instructions](#sox-install)

<a name="citation"></a>
## Citation
If you use our code for research, please cite our paper:
François G. Germain, Qifeng Chen, and Vladlen Koltun. Speech Denoising with Deep Feature Losses. [arXiv:1806.10522](https://arxiv.org/abs/1806.10522). 2018.

### License
The source code is published under the MIT license. See [LICENSE](./LICENCE) for details. In general, you can use the code for any purpose with proper attribution. If you do something interesting with the code, we'll be happy to know. Feel free to contact us.

[Top](#top)

<a name="setup"></a>
## Setup

### Requirement
Required python libraries: Tensorflow with GPU support (>=1.4) + Scipy (>=1.1) + Numpy (>=1.14) + Tqdm (>=4.0.0). To install in your python distribution, run

`pip install -r requirements.txt`

Warning: Make sure your libraries (Cuda, Cudnn,...) are compatible with the tensorflow version you're using or the code will not run.

Required software (for resampling): [SoX](http://sox.sourceforge.net/) ([Installation instructions](#sox-install))

**Important note:** _At the moment, this algorithm requires using **32-bit floating-point** audio files to perform correctly_. You can use sox to convert your file. To convert `audiofile.wav` to 32-bit floating-point audio at 16kHz sampling rate, run:

`sox audiofile.wav -r 16000 -b 32 -e float audiofile-float.wav`

Tested in Ubuntu + Intel i7 CPU + Nvidia Titan X (Pascal) with Cuda (>=8.0) and CuDNN (>=5.0). CPU mode should also work with minor changes.

### Quick start (testing)

For a quick testing, you can download the default validation data by running:

`./download_sedata_onlyval.sh`

followed by:

`python senet_infer.py`

The denoised files will be stored in the folder _dataset/valset_noisy\_denoised/_, with the same name as the corresponding source files in _dataset/valset_noisy/_.

### Default data download

In order to run our algorithm with default parameters, you need to download the noisy dataset from Edinburgh DataShare (see below). The dataset can be automatically downloaded and pre-processed (i.e. resampled at 16kHz) by running the script

`./download_sedata.sh`

To download only the testing data, you can run the reduced script:

`./download_sedata_onlyval.sh`

### Using custom data

If you want to use your own data for _testing_, you need to put all the .wav files in a single folder.

If you want to use your own data for _training_, you need to put your data in a single top folder. In that folder, you should have 4 individual folders:

- _trainset\_noisy/_ (for the noisy speech training files),
- _trainset\_clean/_ (for the ground truth clean speech training files),
- _valset\_noisy/_ (for the noisy validation files), and
- _valset\_clean/_ (for the noisy validation files).

The validation folders may be empty but they must exist. Matching files in the corresponding noisy and clean folders must have the same name.

The audio data *must be sampled at 16kHz* (you can resample your data using SoX - see download\_data.sh for an example).

[Top](#top)

## Denoising scripts <a name="scripts"></a>

### Testing with default parameters

Once you've downloaded in the script download_data.sh, you can directly process the testing dataset by running

`python senet_infer.py`

The denoised files will be stored in the folder _dataset/valset_noisy\_denoised/_, with the same name as the corresponding source files in _dataset/valset_noisy/_.

In our configuration, the algorithm allocates ~5GB of memory on the GPU for training. Running the code as is on GPUs with less memory may fail.

### Testing with custom data and/or denoising model

If you have custom testing data (_formatted as described above_) stored in a folder _foldername/_ and/or a custom denoising model *with names* _se\_model.ckpt.*_ stored in a folder _model\_folder/_, you can test that model on that data by running:

`python senet_infer.py -d folder_name -m model_folder`

The denoised files will be stored in the folder _folder\_name\_denoised/_, with the same name as the corresponding source files.

Warning: At this time, when using a custom model, you must make sure that the system parameters in senet\_infer.py match the ones used in the stored denoising model or the code won't run properly (if running at all).

### Training with default parameters

Once you've downloaded in the script download_data.sh, you can directly train a model using the training dataset by running

`python senet_train.py`

The trained model will be stored in the root folder with the names _se\_model.ckpt.*_.

In our configuration, the algorithm allocates ~5GB of memory on the GPU for training. Running the code as is on GPUs with less memory may fail.

### Training with custome data and/or feature loss model

If you have custom training data (_formatted as described above_) stored in a folder _foldername/_ and/or a custom feature loss model *with names* _loss\_model.ckpt.*_ stored in a folder _loss\_folder/_, you can train a speech denoising model on that data using that feature loss model by running:

`python senet_train.py -d folder_name -l loss_folder -o out_folder`

The trained model will be stored in folder _out\_folder/_ (default is root folder) with the names _se\_model.ckpt.*_.

Warning: At this time, when using a custom loss model, you must make sure that the system parameters in senet\_train.py match the ones used in the stored loss model or the code won't run properly (if running at all).

[Top](#top)

<a name="models"></a>
## Models

The deep feature loss network graph and parameters are stored in the models/loss\_model.ckpt.* files.

The denoising network graph and parameters are stored in the models/se\_model.ckpt.* files.
This model was trained following the procedure described in our associated paper. The current training script se\_train.py is parameterized in such a way that an identical training procedure as in our associated paper would be performed on the specified training dataset.

[Top](#top)

 <a name="data"></a>
## Noisy data

The data used to train and test our system is available publicly on the Edinburgh DataShare website at [https://datashare.is.ed.ac.uk/handle/10283/2791](https://datashare.is.ed.ac.uk/handle/10283/2791). Information on how the dataset is constructed can be found in [Valentini-Botinhao et al., 2016](https://www.research.ed.ac.uk/portal/files/26581510/SSW9_Cassia_1.pdf). The dataset was used without alteration except for resampling at 16kHz.

[Top](#top)

<a name="feature-loss"></a>
## Deep feature loss training

We also provide scripts to (re-)train the loss model. As of know, using the two classification tasks described in our paper is hard-coded.

### Data

Our feature loss network is trained on the _acoustic scene classification_ and _domestic audio tagging_ tasks of the [DCASE 2016 Challenge(https://www.cs.tut.fi/sgn/arg/dcase2016/). Downloading and pre-processing (i.e., downsampling to 16kHz) the corresponding data can be done by running the script:

`./download_lossdata.sh`

Warning: The training script expects the data at the locations set in the downloading script.

[Top](#top)

### Training script

Once the data is downloaded, you can (re-)train a deep feature loss model by running:

`python lossnet_train.py`

The loss model is stored in the root folder by default. A custom output directory for loss model can be specified as:

`python lossnet_train.py -o out_folder`

[Top](#top)

<a name="notes"></a>
## Notes

 * Currently, the download scripts are only provided for UNIX-like systems (Linux & Mac OSX). If you plan on running our algorithm on Windows, please contact us and/or download and resample the data "by hand".

* Currently, dilation for 1-D layers is not properly implemented in the Tensorflow [slim library](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim) we use. The functions _signal\_to\_dilated_ and _dilated\_to\_signal in helper.py allows to transform a 1-D layer into an interlaced 2-D layer such that undilated convolution on the 2-D layer is equivalent to dilated convolution on the 1-D layer.

[Top](#top)

<a name="sox-install"></a>
## SoX installation instructions

The latest version of SoX can be found on their SourceForge page at [https://sourceforge.net/projects/sox/files/sox/](https://sourceforge.net/projects/sox/files/sox/) (Go to the folder corresponding to the latest version). Below are additional details regarding the installations for many common operating systems.

### Linux

#### Ubuntu

As of June 13, 2018, SoX can be installed from the Ubuntu repositories by running in a terminal:

`sudo apt-get install sox`

#### Fedora

As of June 13, 2018, SoX can be installed from the Fedora repositories by running in a terminal:

`sudo yum install sox`

### Mac OSX

#### Homebrew

If you have Homebrew installed, just run in a terminal:

`brew install sox`

#### Macports

If you have Macports installed, just run in a terminal:

`port install sox`

You may need to run the command with root priviledges, in which case, run in a terminal:

`sudo port install sox`

#### Pre-compiled version

SoX provides a pre-compiled executable for Mac OSX. You can download it at [https://sourceforge.net/projects/sox/files/sox/14.4.2/sox-14.4.2-macosx.zip/download](https://sourceforge.net/projects/sox/files/sox/14.4.2/sox-14.4.2-macosx.zip/download).

Then unzip the downloaded archive and move the extracted folder to your _Applications_ folder.

The last step is to add that folder to your path. To do so, run in a terminal:

```
cd ~
echo "" >> .bash_profile
echo "# Adding SoX to path" >> .bash_profile
echo "export PATH=\$PATH:/Applications/sox-14.4.1" >> .bash_profile
source .bash_profile
```

Warning: The executable hasn't been updated since 2015 so consider using one of the two options above instead or compile from sources if the executable fails

### Install from sources (Unix-like systems)

Download sources from the terminal using:

`wget https://sourceforge.net/projects/sox/files/sox/14.4.2/sox-14.4.2.tar.gz/download`

Un-compress the archive:

`tar -zxvf sox-14.4.2.tar.gz`

Go into the folder with the extracted files:

`cd sox-14.4.2`

Compile and install SoX:

```
./configure
make -s
make install
```

Warning: Make sure there are no space in any of the folder name on the path of the source files or the building will fail.

### Windows

Follow instructions provided [here](https://github.com/JoFrhwld/FAVE/wiki/Sox-on-Windows). If you need additional assistance, please contact us.

[Top](#top)
