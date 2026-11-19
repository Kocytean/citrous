### LIME: Explaining Classification with Segmentation
### Citrous: Segmentation by Explaining Classification

_Classification InTegrated ROUghly with Segmentation_: LIME + Rough Set Inference + Uncertainty Quantification (+ Adaptation (in progress))

Train a ResNet18 on the Oxford-IIIT Pet dataset. (A 15 epoch checkpoint with ~73 test accuracy is available [here.](https://drive.google.com/file/d/1e6Xpp_obS_taakMKpFPW8I-ta_ozCaFE/view?usp=drive_link))\
Running the training script will automatically download the dataset; you can stop the training and download the checkpoint to generate the directory structure. Keep the checkpoint in the root folder.

Run `python test_interactive.py` to show progressively developing explanation of the classification over a single image.

Run `python test.py` to evaluate Citrous on 5 images per class from the dataset. Run `python test_lime.py` to evaluate LIME on the same. 
