# Create Contour

In the 2D X-ray Viewport, a contour can be created to mark the outline of a particular part that is depicted on the X-ray. Creating contours can only be done in the 2D viewports and will not work in the project image data set. In the PM tab, a Contour entity will be created as a child entity of the X-ray Object where the contour is created upon. Contours will serve as an input for the Contour-based Registration method in order to match the projected and created contour of the desired part.

When the Create Contour icon ![](Mimics Reference Guide/X-ray/../../Resources/Images/Create%20Contour.png) is clicked, the cursor will change to a pen tool where control points can be placed with a single click. When placing the first point, a line will automatically appear and be attracted to the gradient of the X-ray image. If the contour is not correctly attracted to the border of the desired anatomy, CTRL can be held down to disable the attraction and enable the creation of straight contour sections. When CTRL is released, the contour will be attracted again to the gradient. It is also possible to create spline contours while holding SHIFT during point placement.

![](Mimics Reference Guide/X-ray/../../Resources/Images/CreateContour_Gradient_285x200.png) |  ![](Mimics Reference Guide/X-ray/../../Resources/Images/CreateContour_Straight_285x200.png) |  ![](Mimics Reference Guide/X-ray/../../Resources/Images/CreateContour_Spline_285x200.png)  
---|---|---  
  
The contour can either be closed by clicking on the first control point or left open by double-clicking at the desired end location.

![](Mimics Reference Guide/X-ray/../../Resources/Images/CreateContour_002_480x285.png) |  ![](Mimics Reference Guide/X-ray/../../Resources/Images/CreateContour_ContextMenu%20-01.png)  
---|---  
  
A Contour can exist of multiple segments and does not need to be closed.

![](Mimics Reference Guide/X-ray/../../Resources/Images/CBR_ContourInput_02%20-01_397x241.png)

Exit the Create Contour tool by pressing Esc or clicking the active icon in the X-ray toolbar.

The control points of the Contour can be edited when the tool is not active. Next, an overview of the contour editing actions is given:

Delete point | A control point can be deleted by selecting the point and pressing delete or via the context menu.   
---|---  
Add Point  | Control points can be added to an existing contour by selecting �Add point� in the context menu. When the cursor appearance has changed to ![](Mimics Reference Guide/X-ray/../../Resources/Images/CreateContour_002_table.png) it will create a new control point when clicking on the contour.  
Remove Point | A control point can be removed from a contour by clicking on a control point and selecting �Remove point� in the context menu. The contour will remain closed and will either be attracted to the image gradient or create a linear connection depending on the connection with the adjacent points.  
Split | A contour can be split by selecting a control point and selecting Split from the context menu. The contour will not change visually, but it is now possible to drag the control point to another location.   
Grow | When selecting an end point of a Contour, the Grow functionality enables to extend the contour.   
  
It is possible to change the name and color of the selected contour object by clicking the properties icon in the X-ray Objects tab.

![](Mimics Reference Guide/X-ray/../../Resources/Images/X-ray_ContourProperties.png)

**Recommendations for Contour creation**

Only create contours on locations where the border of the depicted object is clear. X-rays often have areas with low contrast where Contours will not accurately be attracted to the border. In this case, it is better not to draw a contour and skip the area. It is better to use a contour of multiple segments which perfectly match with the border, than a closed contour with parts that don�t fit the border well. |  ![](Mimics Reference Guide/X-ray/../../Resources/Images/CBR_ContourInput%20-%2002_166x284.png)  
---|---  
Place control points in sharp corners to prevent smooth corners.  |  ![](Mimics Reference Guide/X-ray/../../Resources/Images/CreateContour_003_171x186.png) ![](Mimics Reference Guide/X-ray/../../Resources/Images/CreateContour_004_177x188.png)  
It is recommended to draw contours that exceed the contour projection of a particular Part (yellow) if the part is smaller than the created contours.  |  ![](Mimics Reference Guide/X-ray/../../Resources/Images/CreateContour_005_230x292.png)
