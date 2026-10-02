### 3D Interpolate

The '3D Interpolate' allows to edit large regions based on input of marked slices in all 2D viewports . The tool interpolates the marked input slices to create a temporary volume of voxels which can be added, removed, or applied with a threshold to the active mask. 

![](Resources/Images/Menu_Segment/3DInterpolate_myocardium_540x305.png)  
---  
Segmentation of the heart myocardium. The input slices (red) are marked in all viewports and interpolated to create a temporary volume (green).   
  
When the tool is launched, all masks are hidden except the active mask. 

The active mask is marked in blue in the Masks PM tab, 

![](Resources/Images/Menu_Segment/3DInterpolate_DialogBox_Annotations_M20-01_800x150.png)  
---  
(1) Input Type, (2) Input Type Parameters, (3) Mode, (4) Interpolate controls, (5) Operation on Mask (6) Threshold Mode Parameters  
  
#### Input Type

The type defines the shape and characteristics of the editing cursor.

##### Ellipse 

While click and holding the left mouse button, the voxels inside the ellipse will be added, removed or applied with a threshold to the active mask. 

Parameter |  Description |  Shortcut  
---|---|---  
Height  | Length of the vertical ellipse axis. | ctrl + left click + move up/down  
Width | Length of the horizontal ellipse axis. | ctrl + left click + move left/right  
Same Width and Heigth | The cursor will try to preserve a circular shape in regards to the dimensions of the project voxel dimensions.  | -  
  
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

While marking input slices the 'Select' or 'Deselect' mode can be used with any input type.

Mode | Description | Shortcut  
---|---|---  
Select | Add voxels to the input slice.  | D  
Deselect | Remove voxels from the input slice. | E  
  
#### Interpolate

The interpolation of the input slices can be triggered by clicking the 'Interpolate' button ( see picture below). 

Tick the 'Auto-interpolate' checkbox to have the interpolation triggered each time a new input slice is added or existing input slice is modified. 

![](Resources/Images/Menu_Segment/3DInterpolate_DialogBox_interpolate_control.png)  
---  
'Interpolate' control with 'Auto-interpolate' checkbox (top) and 'Interpolate' button (bottom)  
  
#### Operation on mask

Determines the operation that will be done on the active mask. You can choose to remove the voxels in the temporary mask from the active mask, add the voxels in the temporary mask to the active mask or do a local thresholding on the voxels of the active mask, where the voxels of the temporary mask are active.

ModeDescriptionShortcut   
Mode | Description  
Add | Adds the voxels of the interpolated volume to the active mask.  
Remove | Removes the voxels of the interpolated volume from the active mask.   
Threshold | Applies a threshold to the interpolated volume and replaces the intersecting volume of the active mask.
