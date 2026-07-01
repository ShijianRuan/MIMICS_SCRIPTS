## Gradient Threshold  
  
### Introduction

The Gradient Threshold tool allows you to remove regions from a mask where there are strong gradients. This can be useful in several circumstances, for example if you want to generate a mask from MRI that encompasses several muscles and want to largely remove the boundaries between them to create a suitable mask to use in the Muscle Segmentation tool. When you open Gradient Threshold (part of the **Segment > Muscular** menu), the following window will appear.

![](Resources/Images/Gradient%20Threshold/Gradient_threshold_dialogbox_240x234.png)  
---  
Gradient Threshold dialog box  
  
The tool always needs an existing mask ('Input Mask'), from which it will remove strong gradients. With the 'Gradient Threshold' value, you can define how strong the gradients are that should be removed from the 'Input Mask'. If you decrease the value, large parts of the mask will be removed, since even the smallest gradients will be removed by the tool. If you increase the value, the tool will only remove areas where the gradient is very large. The value that is ideal for your specific dataset can vary, with typical values for MRI lying between 5 and 200. See the 'Tips and Tricks' section for more information.

_Output Name - > Add Suffix_

When the Gradient Threshold tool has finished, it will create a new mask in the current project that contains the result. This mask will be named identical to the 'Input Mask'. If desired, the 'Add Suffix' tick box can be ticked, in which case the new mask will get a suffix to its name. In any case, the 'Input Mask' will remain unaffected by the tool.

#### Tips and Tricks

In order to find the most optimal value to set as the Gradient Threshold, there is a trick that you can use. First, go to the Image menu and click on 'Apply Filter'. The following window will open. In this window, add a new filter and choose 'Gradient Magnitude'. This will visualize the strong gradients in your image, and thereby simulate what the Gradient Threshold tool will 'see' when it runs. After clicking 'Ok', you may need to change the windowing of your dataset to achieve an optimal visualization of the filter.

![](Resources/Images/Gradient%20Threshold/gradient_threshold_list_of_filters_360x248.png)  
---  
Filter images dialog box  
  
Here is an example of what an unfiltered and a filtered image look like.

![](Resources/Images/Gradient%20Threshold/gradient_threshold_image_without_filter_240x260.png) |  ![](Resources/Images/Gradient%20Threshold/gradient_threshold_image_with_filter_240x264.png)  
---|---  
Image without filter | Image with filter  
  
Now, if you apply a threshold on this filtered image and slide the minimum threshold value from left to right or vice versa, you can find the ideal value for your dataset. In the case of segmenting muscles from MRI, the value should be set in a way such that as many muscle boundaries as possible are included in the threshold, without including too many other structures. Once you have found your ideal value, you can click 'Cancel' and go back to showing your original images by clicking on 'Show Filtered Images' in the 'Image' menu. Then, you can use that value when you run Gradient Threshold.

![](Resources/Images/Gradient%20Threshold/gradient_threshold_threshold_on_image_with_filter_360x234.png) |  ![](Resources/Images/Gradient%20Threshold/gradient_threshold_image_menu.png)  
---|---  
  
#### Examples of Settings of Gradient Threshold

In the images below, you see (from left to right) the original mask, and the effects of setting a Gradient Threshold value of 10, 25 and 100. As input to the Muscle Segmentation tool, setting the Gradient Threshold at 25 works best for this particular dataset. It is perfectly fine if the muscle mask contains some gaps within its volume, even within muscles. The mask resulting from a value of 10 does not contain a large enough volume of each muscle, whereas most muscle boundaries in the mask resulting from a value of 100 are still included which is not the way it should be as input to the Muscle Segmentation algorithm.

![](Resources/Images/Gradient%20Threshold/gradient_threshold_image_top_left_240x260.png) |  ![](Resources/Images/Gradient%20Threshold/gradient_threshold_image_top_right_240x261.png)  
---|---  
![](Resources/Images/Gradient%20Threshold/gradient_threshold_image_bottom_left_240x260.png) |  ![](Resources/Images/Gradient%20Threshold/gradient_threshold_image_bottom_right_240x262.png)  
  
Note that once you have run the Gradient Threshold tool, it may still be necessary to proceed with using Split Mask and other manual editing tools to obtain a suitable input mask for the Muscle Segmentation tool.
