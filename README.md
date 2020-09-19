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
