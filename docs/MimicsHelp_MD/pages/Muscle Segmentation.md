## Muscle Segmentation

### Introduction

The Muscle Segmentation tool allows you to semi-automatically segment many muscles in one go, rather than segmenting them all one-by-one. This can yield a large reduction in segmentation time. The tool is particularly suitable when dealing with MRI images on which muscle boundaries are well visible, such as in the example shown below. The individual muscles comprising the quadriceps, hamstrings and adductors can be clearly observed in the image on the left. Two inputs are needed to run Muscle Segmentation: a number of previously created atlases (each atlas stored in a separate Mimics project) and one or more input masks that you need to create for the current scan that you want to segment. See the Atlases section for more information on how to create atlases. The ideal input mask that is used as input for the Muscle Segmentation tool would look like the image on the right. It contains all muscle tissue and has some 'gaps' between muscles which will help the algorithm to automatically segment the muscles.

![](Resources/Images/Muscular/muscular_axial_slice_no_mask_360x353.png) |  ![](Resources/Images/Muscular/muscular_axial_slice_with_mask_360x347.png)  
---|---  
  
The Muscle Segmentation algorithm will first register the atlases to the current scan, segment by segment. Next, it determines the best matching atlases. Finally, it creates the individual muscle masks by using a voting mechanism on the basis of the best matching atlases. More details are given below.

### Selecting Masks and Atlases

When you open Muscle Segmentation, the following window will appear:

![](Resources/Images/Muscle%20Segmentation/muscle_segmentation_dialog_box_360x504.png)  
---  
Muscle Segmentation dialog box  
  
The first step is to browse to the folder that contains the atlases that you want to use. Then, select the input masks that you want to use. One input mask should be selected for each segment that the tool should segment. Note that the names of these input masks should match with the mask names in the atlases. See the Atlases section for more information.In this case, the atlases contain masks according to the convention

'upper_right_input_mask',

'upper_left_input_mask',

'upper_right_input_mask_right_tensor_fasciae_latae', 'upper_left_input_mask_right_tensor_fasciae_latae'.

Therefore, the only mask names that can be recognized by the Muscle Segmentation tool are �upper_left�, �upper_right�, �lower_left�, and �lower_right�. In the example shown in the screenshot, the 'upper_right' and 'upper_left' masks are recognized after clicking 'Check Project Compatibility'. 'Correct Mask Naming' will appear in green if the selected masks in the present scan are named correctly.

### Setting the Tool's Parameters

#### Grid Size of Transformation

When the Muscle Segmentation algorithm runs, the atlases are registered to the current scan, segment by segment (e.g. lower leg, upper leg, ...). The 'Grid Size of Transformation' parameter determines how accurately this is done. If a larger value is set, the transformation will use fewer points as input, and as a result, the calculation time will be reduced. A smaller grid size provides a more flexible and accurate transformation. Note that the grid size has a significant influence on the processing time of the algorithm.

#### Sample Percentage

This parameter allows you to set the percentage of voxel samples that will be used to measure the correspondence between each segment and the corresponding atlas segments. If you increase the value, more voxels will be used in this process, leading to a more accurate registration. This parameter has a considerable effect on the processing time. Therefore, caution is advised when increasing this parameter above its default value (1%). Setting this parameter above 50%, and even above 10% is hardly ever needed.

#### Available Atlases

This is a number that is provided for your information, so that you can verify if all the atlases in your atlas folder have been correctly found by the Muscle Segmentation tool.

#### Number of Best Matching Atlases to be used

This parameter allows you to limit the maximum number of matching atlases that are used during the segmentation. The number of atlases that is set here does not have any influence on the calculation time. However, the reason for wanting to limit this number is that in most situations, there will be atlases that are very similar to the current scan (i.e. that have high correspondence, see next parameter), and there will be atlases that are not as similar as the current scan. For example, if there are 10 atlases available, using only the best matching ~3-6 atlases will likely lead to better segmentation results than using all 10 atlases.

Tip: The correspondence values per segment of each atlas are shown in the log panel while the algorithm runs. In this way, you can observe if one or more atlases are performing poorly.

#### Atlas Filtering Threshold

Here you can set a minimum threshold for the correspondence value between an atlas and the current scan. Atlases having a lower correspondence will automatically set to be not used.

#### Minimum Number of Agreeing Atlases

The Muscle Segmentation tool contains a voting mechanism between the atlases with the highest correspondence that determines whether to assign a certain voxel to a certain muscle. This parameter allows you to set how many atlases have to agree. In most cases, setting this parameter to about one third to half of the total number of atlases works best. If there are only very few atlases available, the value can be set to 1 or to the total number of atlases.

#### Output Name -> Add Suffix

When the Muscle Segmentation tool has performed the segmentation, it will create new masks in the current project that contain the newly segmented muscles. If desired, the 'Add Suffix' tick box can be ticked. If this is ticked, when the tool is done, the current Mimics project will be 'saved as' a new Mimics project in the folder where the current Mimics project is stored, with the chosen suffix. The currently opened project will be saved before the algorithm starts and remain unaffected.

After running the Tool: Manual Check and Touch-ups

When the algorithm has finished, it is recommended to inspect the results and to correct manually where needed using the regular mask editing tools or using Contour Editing (after creating a 3D model).

### Atlases

#### Description of TLEMsafe atlases

The Muscle Segmentation tool was created in the European Commission FP7 collaboration project TLEMsafe (Twente Lower Extremity Model � safe, grant agreement no: 247860), of which Materialise was a partner. We gratefully acknowledge the financial support for this project. A set of 10 healthy subject atlases for the lower limb that was created in the course of this project is available upon request from Materialise, through your local application engineer or account manager, or through mimics@materialise.be. For legal reasons, these atlases are in an encrypted format, and can only be used as atlases in the Muscle Segmentation tool. The following collection of screenshots shows what the atlases look like, the muscle masks that are present in the atlases and the naming convention of the masks in the atlases. Note that there are two types of masks present in the atlases: individual muscle masks according to the convention 'upper_right_input_mask_right_tensor_fasciae_latae' as well as larger masks that contain all muscles for a particular segment, according to the convention 'upper_right_input_mask'. The latter mask is very similar to the mask that you should strive to achieve in your own scans to be segmented, to increase the chance of obtaining a good registration between atlas and new scan. The lower limb has been divided into and upper part (hip and thigh) and a lower part (lower leg) because the knee flexion angle during a scan can vary between subjects, and this would negatively affect the registration of the atlases to the scan to be segmented. If you are working with partial leg scans rather than full leg scans (e.g. only the hip area), then the atlases should be cropped to match as well as possible with the input scan. In such a case, please contact your local application engineer or account manager, or mimics@materialise.be, with a brief description of your project and how the atlases should be cropped. You will then be sent cropped atlases rather than full leg atlases.

![](Resources/Images/Muscle%20Segmentation/muscle_segmentation_check_atlas_01_620x240.png)  
---  
![](Resources/Images/Muscle%20Segmentation/muscle_segmentation_check_atlas_02_620x213.png)  
![](Resources/Images/Muscle%20Segmentation/muscle_segmentation_check_atlas_03_620x213.png)  
  
Here is a list of all 39 individual muscle masks per side present in each of the TLEMsafe atlases. Identical 39 masks are present as well for the right limb.

![](Resources/Images/Muscle%20Segmentation/muscle_segmentation_check_atlas_mask_pm_tab_360x385.png)  
---  
  
Here is a table describing the physical characteristics of the atlas subjects. Note that the Muscle Segmentation tool performs scaling during the registration process, therefore manually including or excluding atlases based on height or weight is not needed.

Subject |  Age (y) |  Sex |  Preferred Leg |  Weight (kg) |  Height (mm)  
---|---|---|---|---|---  
1 |  27 |  M |  right |  91,7 |  1800  
2 |  23 |  M |  left |  83,1 |  1820  
3 |  25 |  M |  right |  90,4 |  1950  
4 |  27 |  F |  right |  58 |  1661  
5 |  23 |  F |  right |  77,6 |  1875  
6 |  59 |  F |  right |  64,5 |  1600  
7 |  57 |  F |  right |  55,5 |  1625  
8 |  60 |  F |  right |  70,7 |  1640  
9 |  56 |  M |  right |  75,9 |  1783  
10 |  44 |  M |  right |  81 |  1735  
  
#### Creating your own atlases

Your own atlases should follow a naming convention that is similar to the one used in the TLEMsafe atlases. There should be two types of masks present in the atlases. The first type are individual muscle masks according to the convention 'upper_right_input_mask_right_tensor_fasciae_latae' These masks should follow the contours of the muscle as well as possible, and then be eroded by 1 or 2 pixels, such that they are slightly smaller than the visible contours of the muscle (see screenshot for an example on the psoas muscle).

![](Resources/Images/Muscle%20Segmentation/muscle_segmentation_create_own_atlas_435x396.png)

The second type of masks in atlases are a larger type of masks that are very similar to the input masks that you should strive to achieve in your own scans to be segmented. This is to increase the chance of obtaining a good registration between atlas and new scan. The naming convention should follow this example: 'upper_right_input_mask'. The atlases that you create should be stored as normal Mimics projects (each atlas being a separate Mimics project). If you create additional atlases for the lower limb, you can use a combination of your own atlases and TLEMsafe atlases or use only your own atlases. Materialise does not yet have atlases for other anatomies available. If you would like to collaborate in creating atlases for additional anatomies and make them available to other users, either in encrypted or open format, feel free to contact your local application engineer or account manager, or send an email to [mimics@materialise.be](<mailto:mimics@materialise.be>).

### Frequently Asked Questions

#### Is it faster when segmenting single muscles?

The Muscle Segmentation tool does not provide a benefit when segmenting just one or a few muscles. In those cases, tools like the 3D LiveWire are faster.

#### Does it work on CT?

It will give a result, but the results will not be as good as on MRI, since the muscle boundaries are not visible on CT. If you cannot see the muscle boundaries yourself, neither can the tool.

#### Does it work for segmenting bones from MRI?

It will give a result, but using manual tools like thresholding or 3D LiveWire will likely be faster and lead to better results.

#### Does it work on other anatomies besides muscles?

The tool is optimized for segmenting individual muscles based on one complete initial mask that contains all the muscles. The tool might work on other anatomies when a similar objective is aimed for and if the scan is of sufficient quality, but is unlikely to yield an appreciable benefit compared to other tools (e.g. thresholding, 3D LiveWire, multiple slice edit) in most other circumstances.

#### Do you have a suitable MRI scan protocol?

Refer to Kolk et al. 2015: Muscle Activity during Walking Measured Using 3D MRI Segmentations and [18F]-Fluorodeoxyglucose in Combination with Positron Emission Tomography. Medicine and Science in Sports and Exercise 47(9):1896-1905. Alternatively, the out-of-phase sequence of the DIXON protocol may work as well.
