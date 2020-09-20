# SuperResolution

This is a repository for Single Image (Deep) Super Resolution approaches.

Models Included: SRCNN, Baisic/Advanced DRCNN

Currently the best results are for SRCNN with a Skip connection. Below are a comparison of SRCNN at 400 Epochs on Adam vs SRCNN w/ Skips at 180 Epochs on Adam
![SRCNN](/readme_imgs/srcnn_test_metrics.png)
![SRCNN Outs](/readme_imgs/srcnn_gif.gif)

![SRCNN-Skip](/readme_imgs/test_metrics_skip.png)
![SRCNN-Skip Outs](/readme_imgs/skip_test.gif)
![SRCNN-Skip Outs](/readme_imgs/Tile_Test_Skip.png)

Model_Scripts Folder:
- models.py - file containing all model classes and custom loss functions
- utils.py - all utility functions for things like creating tile dataset, stiching images back together, etc
-(SRCNN_Lab.py, DRC_Basic.py, DRC_Advanced.py) - Model Training/Testing Loops

Image_Scripts Folder:
- Generate_LR_Tiled_Fixed.py - Creates Tiled Dataset for Early Upsample Training
- make_gif.py - used for creating demo gifs
