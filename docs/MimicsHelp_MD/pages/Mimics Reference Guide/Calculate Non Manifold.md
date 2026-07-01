#### Calculate Non-Manifold

For many 3D models, it is necessary to have a common surface between two volumes. As an example consider the different brain regions (white matter, gray matter etc) touching each other, or the ligaments of knee touching the bone or complex structures composed of irregular contacts, namely composites, cortical and trabecular bone, bone and cartilage, etc. These geometries are called Non-Manifold Assemblies which typically contain two or more volumes sharing a common surface. 

Using Mimics Innovation Suite, it is possible to create good quality Non-Manifold volumes starting directly from Masks. Due to good quality of these surfaces you can directly launch go to creating a volume mesh as well without having to bother about surface remeshing or fixing, allowing a straightforward workflow towards FEA. 

The first step for creating a non-manifold assembly is the segmentation of the different materials of the objects being selected. Once the masks are created, add the mask to the assembly by clicking on the space under the Assembly column.

Note: The first mask that is added to the assembly will remain unchangeable. When you add a second mask to the assembly that intersects with the firstly added mask, the intersecting regions will be substituted from the last mask added to the assembly.

![](Mimics Reference Guide/../Resources/Images/non_manifold_assembly_masks.png)

In the FEA/CFD menu select the Create Non-Manifold function. The following 3D dialog will open-up.

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/030001DD_424x428.png)

Here you can select the Pre-smoothing, Matrix reduction and Post smoothing factors. 

  1. Pre-smoothing: This function works on the Mask level and improves the quality of your segmentation. It only works on isolated voxels and thin regions to prevent unwanted geometries in the final volume. As a rule of thumb, if you have a lot of thresholding noise, set this value higher. 
  2. Matrix reduction: The matrix reduction works similar to the one in 3D calculation. It allows you to resample and reduce the size of your final non-manifold. Unless you are working with very high resolution data, you can usually leave this to [1,1]. 
  3. Post-smoothing: The post smoothing detects detailed regions and prevents loss of geometry in those regions. For best results, try with a low value first and increase gradually to obtain the ideal result. You can always smooth your surfaces more in 3-matic if you wish. 


