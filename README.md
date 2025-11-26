# [ECE1508 Final Project] Joint Learning of Exposure Patterns and Stereo Depth from Coded Snapshots

![nikita-reading](https://user-images.githubusercontent.com/37815420/236242052-e72d5605-1ab2-426c-ae8d-5c8a86d5252c.gif)

This project introduces a novel, end-to-end learning approach that jointly addresses two traditionally separate computer vision challenges: Snapshot Compressed Image (SCI) decoding and dynamic stereo depth estimation. The framework is an adaptation of the [DynamicStereo](https://github.com/facebookresearch/dynamic_stereo) repository and was trained using the [DynamicReplica](https://github.com/facebookresearch/dynamic_stereo) dataset.

## Dataset
The [DynamicReplica](https://github.com/facebookresearch/dynamic_stereo) dataset consists of 145200 *stereo* frames (524 videos) with humans and animals in motion. 

### Download the Dynamic Replica dataset
Due to the enormous size of the original dataset, we created the `links_lite.json` file to enable quick testing by downloading just a small portion of the dataset.

```
python ./scripts/download_dynamic_replica.py --link_list_file links_lite.json --download_folder ./dynamic_replica_data --download_splits test train valid real
```

To download the full dataset, please visit [the original site](https://github.com/facebookresearch/dynamic_stereo) created by Meta.

## Installation
To set up and run the project, please follow these steps.

### Setup the root for all source files:
```
git clone https://github.com/kungchuking/E2E_SCSI.git
cd dynamic_stereo
```
### Create a conda env:
```
conda create -n dynamicstereo python=3.8
conda activate dynamicstereo
```
### Install requirements
```
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
pip install -r requirements.txt
```

## Evaluation
To download the checkpoints, you can follow the below instructions:
```
mkdir dynamicstereo_sf_dr
wget -O dynamicstereo_sf_dr/model_dynamic-stereo_010179.pth "https://huggingface.co/kungchuking/E2E_SCSI/resolve/main/dynamicstereo_sf_dr/model_dynamic-stereo_010179.pth"
```
You can also download the checkpoints manually by clicking the [link](https://huggingface.co/kungchuking/E2E_SCSI/resolve/main/dynamicstereo_sf_dr/model_dynamic-stereo_010179.pth). Copy the checkpoints to `./dynamicstereo_sf_dr/`.

For evaluation, see [this notebook](https://github.com/kungchuking/E2E_SCSI/blob/master/notebooks/evaluate.ipynb).

## Training
Training requires a 50GB GPU. You can decrease `image_size` and / or `sample_len` if you don't have enough GPU memory.
However, we chose an `image_size` of 480x640 because it is the resolution of the coded-exposure camera we custom-designed in our lab for research.
If you reduce `sample_length`, your effective compression ratio for SCI is reduced. 
You need to donwload *Dynamic Replica* before training.
If you are on a Linux machine, run `./train.csh` for training:
```
./train.csh
```
Alternatively, you can manually copy and paste the python execution intstruction in the file if your are training on Windows.

## License
[DynamicStereo](https://github.com/facebookresearch/dynamic_stereo) is licensed under CC-BY-NC, however portions of the project are available under separate license terms: [RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo) is licensed under the MIT license, [LoFTR](https://github.com/zju3dv/LoFTR) and [CREStereo](https://github.com/megvii-research/CREStereo) are licensed under the Apache 2.0 license.

