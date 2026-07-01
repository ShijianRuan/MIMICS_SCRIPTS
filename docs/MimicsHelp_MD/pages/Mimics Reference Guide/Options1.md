#### Options  
  
For more information about the Part generation parameters, click the Options button.

Here you can set the parameters for the Custom setting for generating a 3D model. Part visualization is performed by means of triangulation of a segmented 3D area.

Note: The custom parameter selection is remembered for current project, session and in between Mimics sessions. 

The number of triangles determines the quality of the reconstruction: the more triangles, the higher the quality. The downside is that more triangles require more memory. This should be considered when calculating a Part.

Two methods for reducing the number of triangles are available: Image matrix reduction and/or triangle reduction.

A smoothing algorithm changes the overall appearance of the triangular mesh. 

![](Mimics Reference Guide/../Resources/Images/Calculate3DParameters_common_512x474.png)

##### Quality

All these quality aspects are grouped in the predefined Low, Medium and High settings. The Custom setting is user defined. Especially for technical CT applications (and all high resolution scans), it is recommended to study the Part generation parameters and to define practical custom settings.

##### Threshold

###### Mask

This is the regular part generation method that takes into account the threshold values that are defined from the respective mask. The use of this method can cause staircasing effects under conditions. E.g: A mask is created by boolean operation of two other masks. Consequently the assigned threshold does not represent the pixels that are included in the mask. In that case, staircase effect can be noticed after calculating the Part.

###### Voxel

This method is based on calculations of an upper and lower threshold for each voxel. Each threshold has its own �score� which is calculated as amount of included masked voxels minus non-masked voxels. The best thresholds according to this score are selected for this voxel. Then the regular part generation method with different thresholds for each voxel is applied. This method results in general to smoother results.

##### Interpolation methods

###### k. Gray value Interpolation

Gray value interpolation is a real 3D interpolation that takes into account the Partial Volume effect and therefore it is more accurate. With the gray value interpolation method we assume that the bone densities give an indication on the amount of bone within one pixel. All edges of the surface are decided based on the gray values. Also the place in between these 2 pixels is based on the gray values of the 2 pixels.

The advantage of gray value interpolation is that it gives lots of detail and that the dimensions are correct. The disadvantage is that you get unnecessary details due to the noise within the images. With a femur head for instance, you will get better results (meaning: a nicely rounded edge) when using this gray value interpolation. Of course, you will need to do a little smoothing as well in order to reduce the noise.

However, it is important to realize that gray value interpolation does not always produce good results. When the slice distance of the scan deviates considerably from the slice thickness, the resulting mesh gives a noisy surface. Only if the slice thickness and slice distance are the same, gray value interpolation works fine. This condition should be fulfilled during the scanning (acquisition). Changes of the Z resolution (see the paragraph about Matrix Reduction) therefore should not be used in combination with gray value interpolation. A reduction of the XY resolution does not violate the condition.

Gray value interpolation is recommended to use for technical CT applications.

###### l. Contour Interpolation

Contour interpolation is a 2D interpolation in the plane of the images that is smoothly expanded in the third dimension. This interpolation algorithm uses the gray value interpolation within the slices, but in the Z direction a linear interpolation between the contours is used (as schematically represented below). This interpolation method gives the best results for medical purposes.

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/03000135_207x101.png) |  ![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/03000136_207x88.png)  
---|---  
Example of contour interpolation |  Principle of contour interpolation  
  
Conclusion:

A contour interpolation results in a 3D that looks smoother and better (less gaps). Contour interpolation is recommended to use for medical CT applications.

A Gray value interpolation results always in correct dimensions and correct positioning of the 3D, but the 3D can be noisy. Gray value interpolation is recommended to use for technical CT applications.

##### Shell Reduction

This feature is an extra filter which removes small inclusions by only keeping a number - defined by the user - of the largest shells.

##### Slices

By default, the table positions of the first and the last image are shown in the dialog. A default calculation is a calculation of the whole segmentation. 

The calculation can be reduced to a part of the segmentation by adapting the table positions. The Reset button will restore the default values.

##### Smoothing

This function is meant to make rough surfaces smoother. It works like a filter for noise reduction. 

  * The Iteration parameter expresses how many cycles of the smoothing are performed. Don�t exaggerate the number of cycles! All iterations change the triangulation. If too many cycles are passed, every Part will turn into a sphere-like object! The number of iterations defines the area of influence for smoothing.


  * The Smooth factor indicates the importance of local geometry. If this factor is low (close to 0), the local geometry is considered as important and the smoothing is limited. With high values for the ratio (close to 1), the new position is mainly determined by the position of the other points of the triangles in the neighborhood. In this last case it is obvious that we talk about smoothing.


![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/03000137_207x164.png) |  ![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/03000138_207x162.png)  
---|---  
STL generation without smoothing |  STL generation with smoothing  
  
Note: If you take a high smooth factor, the number of iterations should be kept low.

##### Matrix reduction

This option allows grouping of voxels to calculate the triangles. The reduction is given relative to the X-size (= Y-size) of a pixel in the image and relative to the height (Z-size) of a pixel in the 3D data set.

XY resolution |  Decides how many voxels are grouped in the XY plane  
---|---  
Z resolution |  Decides how many voxels are grouped in the Z-direction  
  
An XY- or Z-resolution of 1 means no matrix reduction in the plan or the Z-direction.

Note: When using gray value interpolation, reducing too much will lead to a loss of information for thin or small objects. Artificial holes might appear in the 3D image but the dimensions of the object will stay quite accurate.

Note: When using contour interpolation, reducing too much will lead to incorrect dimensions of the object. The visual representation will be quite good, but when measuring items they will typically appear too large.

If a matrix reduction is set in the plane, you have to option to choose the matrix reduction algorithm, Continuity or Accuracy.

###### m. Continuity algorithm

Using this algorithm for matrix reduction in the XY-plane, will give a very nice result, but the 3D dimensions will become larger when using a bigger matrix reduction.

###### n. Accuracy algorithm

Using this algorithm for matrix reduction in the XY-plane, results in a 3D that is less nice, because there will appear gaps in the surface on places where the wall thickness is smaller than the pixel size after matrix reduction. The positive effect is that the dimensions of the 3D model stay exact.

For the predefined settings (low, medium and high), Mimics automatically selects accuracy for gray value interpolation and continuity for contour interpolation. 

##### Triangle Reduction

Triangle reduction allows you to reduce the number of triangles in the mesh. This makes it easier to manipulate the file. 

There are three Reducing modes of triangle reduction, the Point-type, the Edge-type and the Advanced Edge-type. They all have the same parameters. The advanced edge reduction algorithm generates less noise on the resulting surface and creates a smaller object. The point reduction and the edge reduction are better for technical objects since they create more uniform meshes. 

The Tolerance indicates the maximum deviation in mm that a related triangle may have, to be part of the same plane that contains the selected triangle. It makes sense to keep this value related to the pixel size (e.g. half the pixel size or a quarter). 

The Number of Iterations is a user-set value that defines how many times the program should make the calculations. The algorithm needs several iterations to reduce the number of triangles in larger flat areas. This algorithm converges to a stable result after about 15 iterations. More iterations therefore do not make a lot of sense. 

The Edge Angle-value defines which angle should be used to determine edges of the part that cannot be removed. Triangles deviating less than this angle will be grouped into the plane of the other triangles. 

The Working Buffer size relates directly to the amount of memory used to process the reduction. The higher the buffer size, the greater the part of the segmentation that can be calculated at once. 

It is advisable not to use the reducer on very noisy objects. In this case it is better to perform a smoothing first. 

If a plane, defined by the Tolerance and the Angle consists out of several triangles, the program will try to re-triangulate this area. The Point-type reduction mode will try to reduce the amount of triangles by removing a point. The Edge-type will remove a triangles edge (two points + the connecting line between these two points). If the tolerance-value is too big, essential part information may get lost.
