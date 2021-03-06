# SuperResolution

This is a repository for Single Image (Deep) Super Resolution approaches.

Models Included: SRCNN, Baisic/Advanced DRCNN

Model_Scripts Folder:
- models.py - file containing all model classes and custom loss functions
- utils.py - all utility functions for things like creating tile dataset, stiching images back together, etc
-(SRCNN_Lab.py, DRC_Basic.py, DRC_Advanced.py) - Model Training/Testing Loops

Image_Scripts Folder:
- Generate_LR_Tiled_Fixed.py - Creates Tiled Dataset for Early Upsample Training
- make_gif.py - used for creating demo gifs

Currently the best results are for SRCNN with a Skip connection. Below are a comparison of SRCNN at 400 Epochs on Adam vs SRCNN w/ Skips at 180 Epochs on Adam
![SRCNN](/readme_imgs/srcnn_test_metrics.png)

Gif of Test Results Every 10 Epochs
![SRCNN Outs](/readme_imgs/srcnn_gif.gif)

SRCNN W/ Skip Metrics
![SRCNN-Skip](/readme_imgs/test_metrics_skip.png)

SRCNN W/ Skip Re-Tiled Outputs
![SRCNN-Skip Outs](/readme_imgs/skip_test.gif)

SRCNN W/ Skip Re Tiled Side by Side
![SRCNN-Skip Outs](/readme_imgs/Tile_Test_Skip.png)


