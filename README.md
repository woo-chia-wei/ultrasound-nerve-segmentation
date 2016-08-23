# Our approach to tackle Kaggle Ultrasound Nerve Segmentation competition

More info on this Kaggle competition can be found on [https://www.kaggle.com/c/ultrasound-nerve-segmentation](https://www.kaggle.com/c/ultrasound-nerve-segmentation).

This deep neural network achieved **~0.68 score on the leaderboard (rank: 76/923)**

The architecture was inspired by [U-Net: Convolutional Networks for Biomedical Image Segmentation](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/).
We used the code provided by [Marco Jocic](https://github.com/jocicmarko/ultrasound-nerve-segmentation) as a starting point.

---

## Overview

### Data

[Provided data](https://www.kaggle.com/c/ultrasound-nerve-segmentation/data) is processed by ```data.py``` script.
This script just loads the images and saves them into NumPy binary format files **.npy** for faster loading later.

### Prepare the data

In order to extract raw images and save them to *.npy* files,
you should first prepare its structure. Make sure that ```raw``` dir is located in the root of this project.
Also, the tree of ```raw``` dir must be like:

```
-raw
 |
 ---- train
 |    |
 |    ---- 1_1.tif
 |    |
 |    ---- …
 |
 ---- test
      |
      ---- 1.tif
      |
      ---- …
```

* Now run ```python data.py```.

Running this script will create train and test images and save them to **.npy** files.

### Pre-processing

The images are resized to 64 x 96. Data augmentations such as random rotation, horizontal flip and vertical flip are added at runtime by using ImageDataGenerator module. This is modified version of keras ImageDataGenerator [https://www.kaggle.com/hexietufts/ultrasound-nerve-segmentation/easy-to-use-keras-imagedatagenerator/code](https://www.kaggle.com/hexietufts/ultrasound-nerve-segmentation/easy-to-use-keras-imagedatagenerator/code), as the original version does not augment the masks.

Output images (masks) are scaled to \[0, 1\] interval.

### Model

You can find it [here](http://deepcognition.ai/blog/ultrasound-nerve-segmentation-using-u-net/)

### Train the model and generate masks for test images

* Run ```python train.py``` to train the model.

Check out ```train_predict()``` to modify the number of iterations (epochs), batch size, etc.

After this script finishes, in ```imgs_mask_test.npy``` masks for corresponding images in ```imgs_test.npy```
should be generated. I suggest you examine these masks for getting further insight of your model's performance.

### Generate submission

* Run ```python submission.py``` to generate the submission file ```submission.csv``` for the generated masks.

Check out function ```submission()``` and ```run_length_enc()```

### Post-processing

* Run ```ultrasound_postprocessing.ipynb``` to process the generated ```submission.csv``` to remove masks with width less than 60 pixels.
