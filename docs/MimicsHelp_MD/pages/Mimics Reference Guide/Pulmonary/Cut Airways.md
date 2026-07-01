# Cut Airways

The Cut Airways tool allows cutting the airways at b-level or at the endings starting from a Centerline. The cut airways can be used as input for lung segmentation, to prepare the model for CFD analysis or as a starting point to create a realistic benchtop model.

![](Mimics Reference Guide/Pulmonary/../../Resources/Images/CutAirways1_276x440.png) |  ![](Mimics Reference Guide/Pulmonary/../../Resources/Images/CutAirways2.png) |  ![](Mimics Reference Guide/Pulmonary/../../Resources/Images/CutAirways3.png)  
---|---|---  
  
**Settings**

Dropdown | Make a selection between �Cut lower ending� and �Cut on b-level�  
---|---  
Select Centerline | The cutting planes are automatically created based on a centerline. A centerline can be created using label centerline or using the centerline tool in the Analyze menu.  
Cutting planes | Shows a list of cutting planes. Cutting will fail for the fuchsia colored cutting planes, these planes need to be manually corrected or deleted. Cutting will succeed for green and brown colored cutting planes.  
Delete | Removes the selected cutting plane.  
  
**Adjust the cutting plane**

To update the position of the cutting plane, grab the center of cutting plane and drag it over the centerline.

![](Mimics Reference Guide/Pulmonary/../../Resources/Images/CutAirways4.png) |  ![](Mimics Reference Guide/Pulmonary/../../Resources/Images/CutAirways5.png)  
---|---  
Left-click on the center of the cutting plane and hold the left mouse button. | Drag the center of the cutting plane over the centerline.  
  
To update the orientation of the cutting plane, grab the contour of the cutting plane and move mouse.

![](Mimics Reference Guide/Pulmonary/../../Resources/Images/CutAirways6.png) |  ![](Mimics Reference Guide/Pulmonary/../../Resources/Images/CutAirways7.png)  
---|---  
Left-click on the contour of the cutting plane and hold the left mouse button. |  Move the mouse while holding the left-mouse button to change the cutting plane orientation.
