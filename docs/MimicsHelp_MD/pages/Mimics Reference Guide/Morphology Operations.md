### Morphology Operations

![](Mimics Reference Guide/../Resources/Images/MorphologyOperations.png)

Morphology Operations will perform actions on the 'form' of a mask. The different morphology operations are: 

  * Erode


  * Dilate


  * Open


  * Close


All these functions will take or add pixels from the source mask, the result will be copied in the target mask.

Source |  The Source mask is the mask that will be altered.  
---|---  
Operation |  Erode: erode will take pixels from the edges. Erode followed by a region grow can separate parts.  Dilate: dilate will add pixels from the edges. This can be used to restore the effect of the erosion. You can limit the effect of dilation to another mask. This is to prevent that you will have an end-result that is larger than wanted. Open: will perform first an erosion, immediately followed by a dilation. Small edges will be removed or opened. Close: will perform first a dilation, immediately followed by an erosion. Small cavities will be closed.  
Target |  The new mask that will be created.  
Limited to : |  You can limit the effect of an operation to another mask. This is to prevent that you will have an end-result that is larger/smaller than wanted. Note: When performing an erode action while limiting to the mask, a Boolean intersection is performed.  
Number of pixels |  is the amount of pixels you will take/add in one operation  
8-connectivity  
26-connectivity |  |  ![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/03000117_152x107.png) |  ![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/03000118_158x152.png)  
---|---  
8- connectivity  |  26-connectivity  
  
8-connectivity will only look at the neighboring pixels in the plane. (The operation is performed on the complete dataset)

26-connectivity will look at the neighboring pixels in 3D. However, to take effect, the slice distance must be equal or less than the size defined by the number of pixels (pixel size * number of pixels).  
  
The lower and upper threshold boundaries of the target mask will be taken over from the source mask

After entering the appropriate values, click the Apply button.
