# Threshold Region Growing

The **Threshold Region Growing** tool allow to grow a mask from a selected seed point, all connected voxels within the threshold range and boundary limits will be added to the mask.

![](Mimics Reference Guide/Pulmonary/../../Resources/Images/ThresholdRegionGrowing.png)

Settings

Target |  The marked voxels will be added to the target mask, the user can select an existing mask or create a new mask.  
---|---  
Fill Cavities |  Enables filling internal gaps.  
Multiple Layer | When marked, Threshold Region Growing will be applied on the 3D image stack. When unmarked, the tool will be applied on the slice where the seed point was indicated.  
Threshold |  Only voxels with intensity values within the threshold range will be marked.  
Bounds |  Limits the operation to a user defined cylinder.
