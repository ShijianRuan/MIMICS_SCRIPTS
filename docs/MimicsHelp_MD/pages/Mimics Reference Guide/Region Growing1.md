### Region Grow  
  
![](Mimics Reference Guide/../Resources/Images/region%20growing%201.png)

The Region Growing tool provides the capacity to split the segmentation into separate objects.

Source | The original mask  
---|---  
Target:  | The target mask can be a new mask or an existing object, in which case the selected region will be added to this object.  
Multiple layer | The operation can be performed on one single slice (multiple layer is Off) or in 3D on all slices (Multiple layer is On): to do this, turn Multiple Layer On or Off in the Region Growing Properties toolbar.  
6-connectivity - 26-connectivity |  ![](Mimics Reference Guide/../Resources/Images/kubus_connectivity.jpg) 6-connectivity:will only look at the neighboring faces of the selected voxel 26-connectivity:will look at the neighboring faces, nodes and vertices of the selected voxel  
  
After entering the appropriate values, click the left mouse button (cross shaped) on one point of the object of interest (which has to be part of the current segmentation object.). All points in the current segmentation object that are connected to the marked point will be moved to the target mask.

If two existing masks are chosen in the source and target box, the double arrow can be used to switch source and target. 

When the check "Leave Original mask" is marked, all selected information will be copied and pasted in the new mask. When it is turned off, all selected information will be removed from the source mask and placed in the target mask (compare it to cut and paste).
