# Segment Airway  
  
The **Segment Airway** tool is intended to semi-automatically segment the airway track on inspiration or expiration scans. By indicating the start of the trachea the user initiates the airway segmentation process. The outputs of the tool are a mask and 3D model of the segmented airway.

During the segmentation process it is important to investigate the segmentation for leakages. Leakages occur in regions where the contrast between the airway and the airway wall decreases, in such regions the segmentation can leak into the pulmonary parenchyma and subsequently lung tissue gets erroneously marked as airway. The level of leakages can be controlled with the leakage detection parameter, applying weak leakage detection will result in finding the most airway branches as well as resulting in the most leakages.

The segmentation process will preview the segmentation result in 2D and 3D. Leakages can be removed by placing a leakage indicator on the 3D preview or afterwards by post-processing the mask using edit mask tools.

**General workflow**

  1. Initialization
  2. Manual editing
     1. Add missing branches 
     2. Delete erroneous branches
     3. Indicate leakages
     4. Split leakages
  3. Calculate Part


**Initialization**

The Segment Airways tool allows semi-automatically segmentation of the airway by indicating the trachea.

![](Mimics Reference Guide/Pulmonary/../../Resources/Images/SegmentAirway.png)

**Settings**

Preprocessing | The noise filter should only be used for those subjects where the algorithm fails to start or many branches are missing.  
---|---  
Leakage detection | Weak leakage detection will result in most branches found, the result will contain most leakages. Strong leakage detections will result in a shorter airway and less leakages.  
3D Object post processing | Part post processing will improve the quality of the triangles while maintaining the essential characteristics of the airway; this will facilitate subsequent 3D model based operations.   
  
To initiate the **Airway Segmentation** tool click **Start**. In any 2D view locate the start of the trachea and indicate it by selecting two points.

![](Mimics Reference Guide/Pulmonary/../../Resources/Images/SegmentAirway_Initialize.png)

**Manual Editing**

**Add missing branches**

Missing branches can be added by indicating the start of the missing branch, similarly to how the trachea is indicated. After indicating the missing branch, click on **Continue** in the Segment Airway dialog. Lowering the leakage detection may be needed for the algorithm to detect the missing branch.

![](Mimics Reference Guide/Pulmonary/../../Resources/Images/SegmentAirway_ManualEditing.png)

**Delete erroneous branches**

Erroneous branches can be deleted by selecting the branch part in the 3D preview, Right-Click on the part opens the context menu, now select **Delete** or **Delete with children**.

![](Mimics Reference Guide/Pulmonary/../../Resources/Images/SegmentAirway_DelBranch1.png) |  ![](Mimics Reference Guide/Pulmonary/../../Resources/Images/SegmentAirway_DelBranch2.png)  
---|---  
Right-click to open the context menu, select **Put leakage marker** |  Right-click and select **Delete** or **Delete with Children** to remove the part.  
  
**Indicate Leakage**

Leakages can be resolved by indicating a leakage detector, click in the 3D preview on the leakage, Right-Click to launch the context menu, now select **Put leakage marker**. The leakage marker will resolve the leakage. To remove the leakage marker, select the leakage marker, launch the context menu and select **Delete leakage marker**.

![](Mimics Reference Guide/Pulmonary/../../Resources/Images/IndicateLeakage1.png) |  ![](Mimics Reference Guide/Pulmonary/../../Resources/Images/IndicateLeakage2.png) |  ![](Mimics Reference Guide/Pulmonary/../../Resources/Images/IndicateLeakage3.png)  
---|---|---  
Select part with a leakage |  Right-click to open the context menu, select **Put leakage marker** | The leakage is resolved  
  
**Split Leakage**

An alternative approach to the **Put leakage marker** tool is to split the leakage into subparts and delete to undesired subparts.

Select the part with a leakage and Right-Click to open the context menu, select **Subdivide** in parts from the menu. The leakage will be split into several subparts, select the subparts that can be removed, now select **Delete** from the context menu. To accept the changes, Right-Click on the remaining subparts and select **Accept current setting**.

![](Mimics Reference Guide/Pulmonary/../../Resources/Images/SplitLeakage1.png) |  ![](Mimics Reference Guide/Pulmonary/../../Resources/Images/SplitLeakage2.png)  
---|---  
From the context menu select **Subdivide into parts**. | The part is subdivided into several subparts.  
![](Mimics Reference Guide/Pulmonary/../../Resources/Images/SplitLeakage3.png) |  ![](Mimics Reference Guide/Pulmonary/../../Resources/Images/SplitLeakage4.png)  
Use left-click to select the undesired subparts. The selected parts are indicated in red. From the context menu select **Delete**. |  To accept the changes select **Accept current setting** from the context menu.  
  
**Calculate Part**

Click on calculate Part to calculate the 3D model.

To obtain the same 3D result starting from a mask, the calculate Part and wrap operation should be launched using the following parameters:

**Calculate Part**

Interpolation method: Gray Value

Smoothing

  * Enabled;
  * Smoothing factor 0.7;
  * Iterations 3;
  * Compensate shrinkage ON;


Triangle Reduction Disabled

Matrix reduction: 1x1

_Wrap_

Protect thin walls: Enabled

Closing distance: 0.35

SetSmallestDetail = pixel size
