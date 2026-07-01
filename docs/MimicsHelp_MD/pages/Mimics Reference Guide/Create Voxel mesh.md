#### Create Voxel Mesh

This operation allows you to create a volume mesh based on voxels. 

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/030001EB_510x317.png)

##### Listed masks

Here the created masks are listed. Select the masks from which you want to create a volume mesh.

Note: You cannot calculate a volume mesh for an empty mask

##### Element type

You can choose to create a volume meshed consisting out of Hexahedral or tetrahedral elements.

##### Filtering

Close small holes: Closes small holes.

Filter small parts: Removes small loose parts.

Improve connectivity: Increases the connectivity to neighboring elements.

##### Smoothing

This operation allows decreasing the sharp edges of the voxels. It gives the best results when the voxels are more or less cubic.

Smoothing iteration count: Defines how many times the program should make the calculations.

Smoothing factor: Strength of smoothing � higher values give better smoothing but will change geometry more than smaller ones.

Volume compensation: This feature compensates the shrinkage process associated to the smoothing operation. 

##### Voxel grouping

This option allows grouping of voxels to calculate the volume mesh. The reduction is given relative to the X-size (= Y-size) of a pixel in the image and relative to the height (Z-size) of a pixel in the 3D data set.

XY resolution |  Decides how many voxels are grouped in the XY plane  
---|---  
Z resolution |  Decides how many voxels are grouped in the Z-direction  
  
An XY- or Z-resolution of 1 means no voxel reduction in the plan or the Z-direction.

Apply filtering on grouped voxels: by enabling this option the filters will be applied on the grouped voxels instead of the original ones.

##### Estimated number of elements

Shows the amount of elements that will be created.
