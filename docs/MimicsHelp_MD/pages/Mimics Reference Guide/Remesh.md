#### Remesh

Allows you to select a Part or an STL that will be remeshed. You can select one or more Parts or STLs out of the list. When you click on the OK button, the remesher will automatically be started with the selected object already loaded.

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/030001DE.png)

The Remesh function, performed by 3-Matic, allows to increase the quality of the triangles so that a tetrahedral mesh can be built from them.

In case you are performing the remesh operation to to create surface or volume mesh for FEA/CFD, the following steps should be kept in mind

  * When calculating a 3D in Mimics, you can check the the shell reduction option so that some very small shells are removed. The option can be accessed in the Calculate 3D dialog by chosing Options. In that dialog you can choose to do a Shell reduction with value 1.


  * To fill out holes and create a continuous volume, you can use the Fill Cavity function from Polyline tool to remove holes from your Part or use the Wrap function in the Remesher.


  * When converting image data to FEA meshes, a lot of unwanted detail can be segmented along with your main body. The morphology operation in Mimics can help you reduce this detail on a mask level. 


##### Remeshing Protocol

Below you can find a basic remeshing protocol. These are the steps that are normally taken to make sure the mesh is optimal for FEA purposes. The different parameters for each step are dependent on each dataset and won't be given in this basic protocol, but an example of the application of this protocol can be found in Tutorial Case 8. 

STEP A: Check if the Part doesn�t have thin walls or structures that will be hard to model but are of no importance for your FE Analysis. You can use the Wrap function to create a closed enveloping surface around your part and remove all the small inclusions that may have resulted from the segmentation.

STEP B: Depending on the parameters you�ve selected for 3D calculation, the resulting Part may include too much detail. As a first step you can do a defeaturing on the Part therefore select the Smooth![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/030001DF.png) button from the Fix toolbar. Next apply Reduce![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/020001E0.jpg) to reduce the number of triangles. 

STEP C: Check the size of the triangles. To do this, create an inspection scene by selecting

your object and clicking on![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/030001E1.png).Select from the inspection measure dropdown box Smallest edge length (A).

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/030001E2_359x62.png)

In the histogram, select as current measure the Inspection measure and check the mean value of the smallest edge length. Do exactly the same for the largest edge length. Remember or write down these values.

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/030001E3_409x170.png)

STEP D: When you notice in the inspection diagram that there are still too many small triangles, you can filter them out with the Filter Small Edges![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/030001E4.png) button which can be found in the Fix toolbar.

STEP E: After you�ve reduced the amount of triangles you should optimize your mesh with the Adaptive Remesh![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/020001E5.jpg) tool. First select the desired quality parameter from the quality Shape measure dropdown box. Typically for FE analysis the Height/Based (N) parameter is used, while for CFD the Skewness (N) parameter is used. With the Auto Remesh function, a quality threshold of 0.3 to 0.4 can be achieved.

STEP F: If the 3D model consists of a very large number of traingels, you can reduce the number of triangles preserving the achieved quality with the Quality Preserving Reduce Triangles![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/030001E6.png) function.

STEP G: In some cases there are still some low-quality triangles left at this point (you should not have started any local operations already). They are usually removed by another call to the split-based algorithm (with larger geometric error than before). 

STEP H: If low-quality triangles still persist, use local operations to fix them.

STEP I: Make the mesh more uniform by doing several calls to the Quality Preserving Reduce Triangles with increasing geometric error. Stop when both the mesh looks uniform enough and the total number of triangles is small enough. 

STEP J: The self-intersection test with the Mark Intersecting Triangles tool and fix them by deleting the appropriate triangles and filling the resulting hole. 

STEP K: If small triangles persist and you do not want to increase the geometrical error any further, use the grouping around 'smallest edge length' to collapse them manually. 

STEP L: Check and remove sharp geometry using the sharp geometry measures (sharp geometry, peaks, shafts) which can be found in the inspection measures. You can do this step based on intuition or on the feedback from the created volume mesh quality from your FEA-preprocessor. 

STEP M: If needed you can control the growth of your triangles with the Growth Control tool ![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/020001E7.jpg).

STEP N: If steps G, H and/or I introduced low-quality triangles call the Auto Remesh algorithm again. 

STEP O: Check if there are large triangles compared to the local wall thickness by using the inspection measure Wall thickness/Edge length (A). You should have at least 4 elements in a wall for CFD applications and minimum of 1 or 2 elements for FE analysis.

STEP P:Create Volume Mesh by clicking on the corresponding icon ![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/020001E8.jpg). Select the method you want to use for the calculation and specify the maximum edge length as having the same value as specified for the Adaptive remesh tool (if applicable). In case you are working with non-manifold assemblies created directly from masks, you can group the sub-volumes according to the original mask. This means that each sub-volume will correspond to one material, allowing a straightforward process during material assignment. Select the preferred shape measure and specify the shape quality threshold desired for your volume mesh.

![](Mimics Reference Guide/../Resources/Images/volume_remesh_new.png)

STEP Q:Analyze Mesh Quality by clicking on the icon ![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/020001EA_25x27.jpg). Select the checkbox for Analyze volume mesh and select the adequate shape measure. Indicate the shape quality threshold value and specify the histogram interval for your analysis. In case you want to visualize the surface triangles in the area of the bad elements, select Mark bad triangles and specify the element growth. 

STEP R: In case you have some low quality elements, go back to STEP I and remesh your surface mesh. Repeat the steps O and P until you obtain a volume mesh with the desired quality.

STEP S: If your volume mesh has an adequate quality for your FE analysis, close the Remesher and export your mesh with the desired file format.

For more information on each of the above-mentioned tools, consult the 3-matic Reference Guide.
