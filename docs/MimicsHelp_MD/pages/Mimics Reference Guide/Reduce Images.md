#### Reduce Images

High-resolution images can contain a lot of noise and this means that a noisy 3D is obtained. When you want to filter the noise within the images or want to speed up the segmentation functions, use a voxel reduction. The Reduce Images window (see picture) is shown when loading data sets that require more memory than available or when the preference setting �Always ask to reduce images when loading� (on the General tab page) is selected. Based on the amount of images and the pixel size, the necessary amount of memory is calculated and compared with the total amount of memory (RAM). If more memory is needed than available, a reduction is proposed. 

![](Mimics Reference Guide/../Resources/Images/ReduceImages.png)

This function groups voxels together when loading the study. A reduction value of 1 means no reduction; every single voxel will be loaded. A reduction of 2 will group 4 voxel together (2 in the X-direction and 2 in the Y-direction) as 1 voxel with as gray value the mean gray value of the 4 voxels. The maximum allowed reduction is 5. Since pixels are grouped together, the noise is filtered out and all segmentation tools work faster.
