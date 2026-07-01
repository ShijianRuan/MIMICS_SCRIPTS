## Introduction

The Muscle Segmentation tool that is part of the Segment -> Muscular menu allows you to semi-automatically segment many muscles in one go, rather than segmenting them all one-by-one. This can yield a large reduction in segmentation time. As input, the Muscle Segmentation tool needs one or more input masks that you need to create for the current scan that you want to segment. The ideal input mask would look like the image on the right. It contains all muscle tissue and has some 'gaps' between muscles which will help the algorithm to automatically segment the muscles. Before you start creating your input mask, make sure your project has been sliced with a consistent thickness as this is a requirement for the Muscle Segmentation algorithm. You can check this by clicking on �Organize Images� in the Image menu and scrolling through the slices. If the slice increment is inconsistent, reslice your project before using Muscle Segmentation. Very small differences like 0.999 and 1.001 are no problem for the tool.

![](Resources/Images/Muscular/muscular_axial_slice_no_mask_360x353.png) |  ![](Resources/Images/Muscular/muscular_axial_slice_with_mask_360x347.png)  
---|---  
  
The **Advanced Segment > Muscular** menu offers a typical sequence for creating such an input mask.

![](Resources/Images/Muscular/muscular_menu_items_164x87.png)  
---  
Muscular menu items  
  
Starting with 'New Mask', you should select an area that encompasses all muscle tissue.

![](Resources/Images/Muscular/muscular_set_threshold_axial_slice_277x300.png) |  ![](Resources/Images/Muscular/muscular_threshold_dialogbox_360x326.png)  
---|---  
  
For help in using the next step, 'Gradient Threshold', [see this section in the help file.](<GradientThreshold.md>)

After running Gradient Threshold, use 'Split Mask', 'Region Growing' and/or 'Edit Masks' to remove parts of the mask that are not muscle tissue, such as the large section in the bottom left of the image below.

![](Resources/Images/Muscular/muscular_threshold_mask_result_240x231.png)  
---  
Result of the Gradient threshold.   
  
Once your mask looks like shown in the example at the beginning of this help file, you can proceed with running Muscle Segmentation, [see this section in the help file.](<Muscle%20Segmentation.md>)
