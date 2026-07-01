### Smart Expand  
  
The Smart expand dilates a rough initial mask until it meets the contours of the desired anatomy. It will add pixels from the edges of a given Source Mask. However, the Smart Expand detects the gray value gradients in the images and limits the growth of the source mask to these gradients. In other words, it stops expanding when it finds an edge in the image. The Smart expand is ideal to create an initial mask in those cases where a normal threshold fails to well delineate the anatomy. It can as well be used to close holes in a mask.

![](Mimics Reference Guide/../Resources/Images/smart%20expand%201.png)

Source Mask |  The Source mask is the initial mask that will be expanded. It will expand till it finds an edge in the images according to Gray value gradients. To learn how to create a good Source mask so that all the desired regions are detected, see the tutorial on Smart Expand.  
---|---  
Target Mask  |  The Target mask is the resulting mask. You can chose to create a new mask as the output or replace an existing mask with the output of the algorithm.  
Maximum Expand Distance |  Maximum expand distance defines the region in which the algorithm searches for an edge. Note that larger the expand distance, more time the algorithm takes to run.
