### Edit Masks  
  
The Edit Masks tool allows manual editing operations on 2D images as well as on the 3D preview of the active mask. 

![](Mimics Reference Guide/../Resources/Images/Menu_Segment/EditMasks_DialogBox_Annotations_M20_72_1dpi.png)  
---  
(1) Input Type, (2) Input Type Parameters, (3) Mode, (4) Threshold Mode Parameters, (5) Crop Box toggle button, (6) Update Polylines, (7) Toggle cursor preview.   
  
#### Input Type

The type defines the shape and characteristics of the editing cursor.

##### Ellipse 

While click and holding the left mouse button, the voxels inside the ellipse will be added, removed or applied with a threshold to the active mask. 

Parameter |  Description |  Shortcut  
---|---|---  
Height  | Length of the vertical ellipse axis. | ctrl + left click + move up/down  
Width | Length of the horizontal ellipse axis. | ctrl + left click + move left/right  
Same Width and Heigth | The cursor will try to preserve a circular shape in regards to the dimensions of the project voxel dimensions.  | -  
  
i |  It is not possible to edit the mask in 3D if the visualization of 3D mask preview is switched OFF.  
---|---  
  
  


##### Rectangle

While click and holding the left mouse button, the voxels inside the rectangle will be added, removed or applied with a threshold to the active mask. 

Parameter |  Description |  Shortcut  
---|---|---  
Height  | Height of the rectangle. | ctrl + left click + move up/down  
Width | Width of the rectangle.  | ctrl + left click + move left/right  
Same Width and Heigth | The cursor will try to preserve a square shape in regards to the dimensions of the project voxel dimensions.  | -  
  
##### Lasso

The lasso type allows you to draw a freeform shape on the viewports. Hold the left mouse button while outlining the region of interest and release to perform the action. The area within the freeform line will be applied with the selected mode to the active mask. 

##### Flood-Fill

To use the Flood-Fill tool: select Flood-Fill as Type, select a point in the area you are interested in selecting and move your mouse keeping the left mouse button pressed. The selected area grows until an edge in the image is detected

Parameter |  Description  
---|---  
Tolerance | The tolerance controls the edge detection in the image, in order to stop the contour from growing outside the limits. The higher the value, the sharper the image edge has to be to stop the contour growth.   
Impatience | The impatience represents the speed of expansion out of current region. When the contour approaches an edge, the speed of expansion will gradually decrease.  
  
##### LiveWire

Start indicating some points at the boundaries of the part of interest. A line is created between the points and snaps to the contours of the object. The line indicates the region where the selected operation will be applied.

Parameter |  Description  
---|---  
Gradient Magnitude | This parameter indicates to which kind of gradients the contour will be attracted. If the selected value is close to 0, the contour will be attracted to darker regions on the boundaries of the object. If the selected value is close to 1, the contour will be attracted to brighter regions lying at the boundary of the object/   
Attraction | The attraction coefficient indicates if some cavities in the boundaries of the object should be taken into account or should be neglected. If a value near to -3 is selected, all the small inclusions will be included in the mask. If a value near to 3 is selected, the inclusions will be excluded from the automatic contour.   
  
#### Mode

Mode |  Description  
---|---  
Add | Adds voxels to the active mask   
Remove | Removes voxels from the active mask.   
Erase a full slice via the context menu (right-click), and select 'Erase full slice'.   
Threshold | Applies a threshold to within the cursor defined region.   
  
##### Threshold settings

When the threshold mode is active, the last used preset will be active. There are 3 presets: 'Mask', 'Custom 1' and 'Custom 2'. The 'mask' preset reads and sets the threshold range from the active mask. The 'Custom 1' and 'Custom 2' preset are user defined. To set a custom preset range, click on the 'Custom 1' or 'Custom 2' button and change the threshold range. The custom preset values will be remembered when the tool is closed. 

![](Mimics Reference Guide/../Resources/Images/Menu_Segment/EditMasks_ThresholdSettings.png)  
---  
Threshold mode settings with minimum and maximum threshold values and preset buttons.   
  
i | Use the threshold shortcut to easily set a threshold range.   
---|---  
| Minimum threshold value:  |  ![](Mimics Reference Guide/../Resources/Images/Menu_Segment/threshold_tooltip_minimum_threshold.png)  
| Maximum threshold value: |  ![](Mimics Reference Guide/../Resources/Images/Menu_Segment/threshold_tooltip_maximum_threshold.png)  
  
The threshold range will be applied the marked area, regardless of whether the area was already included in the mask or not. While voxels within the range are added to the mask, voxels that lay outside the threshold range will be removed from the mask. 

#### Update Polylines

After editing, press the 'Update Polylines' button to update the polylines that are linked to the active mask.

i | You can still access the 1-click navigation function by pressing the SHIFT button while you are editing. You can then click with your left mouse button on the point you want to navigate to.  
---|---
