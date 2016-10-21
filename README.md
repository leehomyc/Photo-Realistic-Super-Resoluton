# Photo-Realistic-Super-Resoluton
Torch Implementation of "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network"
[[Paper]](https://arxiv.org/pdf/1609.04802)

This is a prototype implementation developed by [Harry Yang](https://scholar.google.com/citations?user=jpIFgToAAAAJ&hl=en&authuser=3). 

<img src='pics/input.png' width=160> <img src='pics/output.png' width=160>

## Getting started

####Training
prepare your images under a sub-folder of a root folder
``` bash
train_folder=your_root_folder model_folder=your_save_folder th run_sr.lua 
```

<<<<<<< HEAD
By default it resizes the images to 96x96 as ground truth and 24x24 as input, but you can specify the size using `loadSize`. Note current generator network only supports 4x super-resolution.
=======
By default it resizes the images to 96x96 as ground truth and 24x24 as input, but you can specify the size using `loadSize` and `scale`.
>>>>>>> 2e06dcd73670f06c9160101f0eb9753a6e81d841

####Loading a saved model to train
```
D_path=your_saved_D_model G_path=your_saved_G_model train_folder=your_root_folder model_folder=your_save_folder th run_resume.lua
```

####Testing
prepare your test images under a sub-folder of a root folder
```
test_folder=your_root_folder model_file=your_G_model result_path=location_to_save_results th run_test.lua
```

## Report Issues
[Contact](mailto:harryyang.hk@gmail.com)

##Todo:

1. Add TV loss.
2. Provide trained models.
