# [ECE1508 Final Project] Joint Learning of Exposure Patterns and Stereo Depth from Coded Snapshots

![Overview](images/overview.gif)

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
To download the pre-trained model weights (checkpoints), please follow the instructions below.

### Command Line Download
You can use the following commands to create the required directory and download the primary checkpoint directly from the Hugging Face repository:
```
mkdir dynamicstereo_sf_dr
wget -O dynamicstereo_sf_dr/model_dynamic-stereo_050895.pth "https://huggingface.co/kungchuking/E2E_SCSI/resolve/main/dynamicstereo_sf_dr/model_dynamic-stereo_050895.pth"
```
### Manual Download
Alternatively, you can manually download the checkpoints by clicking the [link](https://huggingface.co/kungchuking/E2E_SCSI/resolve/main/dynamicstereo_sf_dr/model_dynamic-stereo_050895.pth). Ensure the downloaded file is placed in the required path: `./dynamicstereo_sf_dr/`.

### Evaluation Notebook
For detailed instructions on how to evaluate the model, please refer to the dedicated [evaluation notebook](https://huggingface.co/kungchuking/E2E_SCSI/blob/main/notebooks/evaluate.ipynb).

### Evaluation and Validation
To execute the final evaluation on the DynamicReplica test set, navigate to the `evaluation`directory and run the following Python script:
```
cd evaluation
python evaluate.py
```

## Training
### Hardware and Memory Requirements
Training the model requires a minimum of a 50GB GPU.
* **Memory Adjustment**: If your GPU memory is limited, you may decrease the `image_size` and/or the `sample_len` parameters.
* **Resolution Note**: The chosen `image_size` of 480x640 corresponds to the native resolution of the custom-designed coded-exposure camera used for our research.
* **Compression Impact**: Reducing the `sample_length` will inherently decrease the effective compression ratio for the Snapshot Compressed Imaging (SCI) process.
Before starting training, you must download the Dynamic Replica dataset.
### Execution
If you are running on a Linux machine, use the provided shell script for training:
```
./train.csh
```
For other operating systems, you can open the `./train.csh` file and manually copy and execute the instruction.

## License
Portions of the project are available under separate license terms: [DynamicStereo](https://github.com/facebookresearch/dynamic_stereo) is licensed under CC-BY-NC, [RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo) is licensed under the MIT license, [LoFTR](https://github.com/zju3dv/LoFTR) and [CREStereo](https://github.com/megvii-research/CREStereo) are licensed under the Apache 2.0 license.

