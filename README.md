# DLMI_Proj
Depth Anything V2 in Medical Imaging

This repo conatians the code for the evlauation process of Depth Anything V2 on the EdsoSLAM and Hamlyn Dataset
Running Depth Anything V1/V2 is a pre-contidion for running this code.

Depth Scaling is applied per image in order to ensure consistency with other methods by using the ratio of ground truth to prediction median values

Hamlym dataset Evaluation:
Depth estimation for the Hamlym dataset was calculated by the metric Depth Anything V2 large model, the result of the MDE process are stored as numpy arrays (*..npy), Hamlyn stores true depth as unit16 *.png images. 
The saturation depth is set to 300 [mm] with pixel values lower that 1mm ignored.

EndoSLAM dataset Evaluation:
Depth estimation for the EndoSLAM dataset was calculated by the image-base Depth Anything V2 large model, EndoSLAM stores results are stored as unit8 (320,320,4) png images.

  - True depth must be stored in true_depth_path
  - Output of Depth Anything V2 must be stored in dav2_path
Also added is the code for the Visualization in the paper. 
