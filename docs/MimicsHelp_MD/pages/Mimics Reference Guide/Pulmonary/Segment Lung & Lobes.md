# Segment Lung & Lobes

The �**Segment Lung & Lobes**� segments the lungs and detects the lung separating fissures; subsequently the lungs are cut into lung lobes. The operation creates masks and 3D models of the left and right lung, as well as separate 3D models for each of the lung lobes.

**General workflow**

  1. Initialization
  2. Adjust Fissures
  3. Cut lungs into lung lobes


**Initialization**

To initiate the �**Segment Lung & Lobes**� tool an airway centerline model is required, for consistent results it is advised to use an airway model cut at b-level. More information on how to create an airway model cut at b-level read the sections on Segment Airway, Label airway and Cut Airway.

![](Mimics Reference Guide/Pulmonary/../../Resources/Images/SegementLung&Lobes.png) |  ![](Mimics Reference Guide/Pulmonary/../../Resources/Images/SegementLung&Lobes2.png)  
---|---  
To initiate lung & lung lobe segmentation select an airway centerline and click **Next**. |  Airways cut at b-level can be created using the Segment airway, Label airway and cut airway tool.  
  
**Advanced Settings**

Lower Threshold | Defines the lower threshold for the initial mask of the lung.  
---|---  
Higher Threshold | Defines the upper threshold of the initial mask of the lung.  
  
**Adjust fissures**

The fissures are automatically detected. To adjust the fissure, click on **Edit Fissures**.

![](Mimics Reference Guide/Pulmonary/../../Resources/Images/SegementLung&Lobes3.png) |  ![](Mimics Reference Guide/Pulmonary/../../Resources/Images/SegementLung&Lobes4.png)  
---|---  
To adjust the fissures, click **Edit Fissures** |  The fissures are visualized in the 2D and 3D views. To toggle the visibility, select the glasses in the visible or contour visible column.  
  
**Add point**

To add a point to a fissure, select a fissure from the list and click on **Add Point**. A point can be added in the 2D or 3D view.

![](Mimics Reference Guide/Pulmonary/../../Resources/Images/SegementLung&Lobes5_203x309.png) |  ![](Mimics Reference Guide/Pulmonary/../../Resources/Images/SegementLung&Lobes6.png) |  ![](Mimics Reference Guide/Pulmonary/../../Resources/Images/SegementLung&Lobes7.png)  
---|---|---  
Select a fissure in the list and click on **Add Point**. | Indicate the point in the 2D or 3D view. | The fissure is update.  
  
**Remove point**

To remove a point from a fissure, select a fissure from the list, select a point of the fissure in the 2D or 3D view and click on **Remove Point**. The fissure is updated based on the remaining points. To select multiple points at once, hold shift and the left mouse bottom, drag a box over the points you want to select.

![](Mimics Reference Guide/Pulmonary/../../Resources/Images/SegementLung&Lobes8.png) |  ![](Mimics Reference Guide/Pulmonary/../../Resources/Images/SegementLung&Lobes9_159x243.png) |  ![](Mimics Reference Guide/Pulmonary/../../Resources/Images/SegementLung&Lobes10.png)  
---|---|---  
Select a point in the 2D or 3D view. The center of the point becomes green. |  Click on **Remove Point** to delete the point from the fissure. | The fissure is update.  
  
**Assign to**

The �**Assign to** � tool, allows changing a set of points from the right oblique fissure to the left oblique fissure or vice versa. First select one point using left-click or select a set of points by holding SHIFT and the LEFT mouse button, and drag the mouse over a set of points. Finally click on �Assign to� and select from the menu the fissure you want to assign points to.

![](Mimics Reference Guide/Pulmonary/../../Resources/Images/SegementLung&Lobes11.png) |  ![](Mimics Reference Guide/Pulmonary/../../Resources/Images/SegementLung&Lobes12_141x216.png) |  ![](Mimics Reference Guide/Pulmonary/../../Resources/Images/SegementLung&Lobes13.png)  
---|---|---  
Select a point in the 2D or 3D view. The center of the point becomes green. |  Click on **Assign to** , select the fissure you want to re-assign the points to. | The fissure is update.  
  
**Cut lung into lobes**

In a last step the 3D models of the lung are cut into the different lung lobes.

![](Mimics Reference Guide/Pulmonary/../../Resources/Images/SegementLung&Lobes14.png) |  ![](Mimics Reference Guide/Pulmonary/../../Resources/Images/SegementLung&Lobes15.png)  
---|---  
Freeform surfaces following the lung lobe separating fissures. | The resulting 3D models of the separate lung lobes.
