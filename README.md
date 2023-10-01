## Automatic segmentation framework of X-Ray tomography data for multi-phase rock using Swin Transformer approach

### Prerequisities
The neural network is developed with the mmseg library, we refer to the [mmsegmentation](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/get_started.md#installation) for the installation.

This code has been tested with mmcv 2.0.1 and mmseg 1.1.1, using PyTorch as backend.

The following dependencies are needed:
- numpy
- PIL
- opencv
- tqdm

Also, you will need the coal sample database, which can be freely downloaded as explained in the next section.

### Training

First of all, you need the database. You can get the coal sample dataset from [here](https://figshare.com/projects/Raw_data_and_segmented_label_data_of_rock_sample_derived_from_X-ray_CT/162046). Extract the images to a folder. This folder should have the following tree:
```
Data
└───images
|    ├───training
|    └───validation
└───annotations
    ├───training
    └───validation
```

**[training settings]**  
Here you can specify:  
- *dataset_path*: the input dataset path.
- *save_path*: output path for saving training log and network weights.
- *model*: you can choice UNet/ResNet-DeeplabV3+/SwinTransformer model with unet/deeplab/swin for training, default swin.
- *iteration_time*: number of training iteration.
- *label_class*: the number of class in label, default 4 in coal sample dataset.
- *patch_size*: the input size during training, default 448*448.
- *batch_size*: mini batch size.
- *auxiliary_head*: auxiliary network to help network training.

After all the parameters have been configured, you can train the neural network with:
```
python train.py -dp *dataset_path* -sp *save_path*
```
The 'dp' and 'sp' presenting 'dataset_path' and 'save_path' are required and others are optical.

If available, a GPU will be used. 

