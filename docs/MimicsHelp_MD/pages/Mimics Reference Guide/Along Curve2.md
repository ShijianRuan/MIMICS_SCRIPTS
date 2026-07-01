#### Along Curve

The reslice along curve function allows you to perform a CPR (curved planar reconstruction). As a result, the cross-section view perpendicular to the curve, and the straightened view are given, together with the Part.

![](Mimics Reference Guide/../Resources/Images/Along%20Curve/02000007_605x387.jpg)

By scrolling, the straightened view can be rotated around its axis (i.e. the curve), which is visualized by the yellow line in the cross-sectional view. In this cross-sectional view, you can scroll along the curve. This is visualized in the straightened view by the red line. Moreover, if the reference planes are toggled on in the 3D view, the cross-sectional view is shown as reference on the Part. 

It is possible to:

  * Select an already created curve: a window is shown in which all possible curves are listed


  * Select an already created spline or centerline from the dialog window which is shown.
  * Make sure that this curve�s visibility is toggled on. As a result, the control points of that curve will become visible.
  * Click the first point, and double click the second point. In the straightened view, the first clicked point will be on top.


  * Create a new curve: the cursor will change to a pencil ![](Mimics Reference Guide/../Resources/Images/Along%20Curve/03000004.png). The points that define the curve can be indicated in all 2D views, or on the Part. While placing the points, you can use the interactive MPR. In this way, it is possible to reslice the stack of images as to obtain the desired views for correct placing of the spline points.
  * Panoramic: the axial view is zoomed to full screen, and the cursor will change to a pencil ![](Mimics Reference Guide/../Resources/Images/Along%20Curve/03000004.png). The points that define the curve can be indicated in the axial view. The last point is indicated by a double click. The Reslice Curves toolbox will appear on the screen with the second button already selected (see below).  
The amount of cross-sectional grids can be changed via **Edit > Preferences,** and choosing the Reslicing section.


![](Mimics Reference Guide/../Resources/Images/Along%20Curve/02000008_581x370.jpg)

Note 1: STLs are only visible in the parallel images when a straight reslice curve is drawn.

Note 2: When you left-click once when zooming on a cross-sectional image, that image gets enlarged. By clicking with the unzoom tool on the enlarged cross-sectional image, you will see the original view again.

. 

For the first two options (select and create curve), the CPR is added to the Reslice Objects tab as CPR curve. For the last option (Panoramic), the CPR is added to the Reslice Objects tab as Reslice Curve (RC).

##### Reslice Objects tab

Several functions on the objects can be accessed via the Reslice Objects tab in Project Management area.

![](Mimics Reference Guide/../Resources/Images/ResliceObjects_OptionsTab.png)

New |  Creates a new reslice object. The different options appear: ![](Mimics Reference Guide/../Resources/Images/AlongCurve_options.png) Click on the desired option  
---|---  
Delete |  Deletes the selected object  
Properties |  Displays the properties of the selected object  
Duplicate |  Duplicates the selected object  
Actions |  The action button lists all the available actions on the Reslice objects. ![](Mimics Reference Guide/../Resources/Images/ResliceObjects_options_AlongCurve.png)  
  
##### Properties dialog for CPR curves (Select and Create Curve option)

![](Mimics Reference Guide/../Resources/Images/CPR_Properties.png)

Name |  Change the name of the CPR.  
---|---  
Rotation angle step |  Change the rotation angle step of the rotation of the straightened view around the axis (i.e. the curve) between the limits provided.  
Pixel size |  Change the pixel size of the resliced views (default size is the project's pixel size).  
Frame size |  Change the frame size of the straightened view (the width). The cross-sectional view size will change accordingly.   
Help | Opens the respective help chapter from the help file.  
  
##### Reslice Curves Properties

The Reslice Curves toolbox can be accessed via the Reslice Objects tab in the Project Management. Select the reslice object and click on the Properties button. 

![](Mimics Reference Guide/../Resources/Images/ResliceCurves_Properties.png)

Name | Change the name of the reslice curve  
---|---  
Modify

> ![](Mimics Reference Guide/../Resources/Images/ResliceCurves_AddPoint.png) Add Point ![](Mimics Reference Guide/../Resources/Images/ResliceCurves_RemovePoint.png) Remove Point

|  Add or remove points on the reslice curve. Click on the Add point button to add a point in the middle of the selected line. When you hover the mouse above the reslice curve, the cursor will change into a pencil. Click with the left mouse button to add a point. In order to exclude a point from the reslice curve click on the point of the reslice curve you want to delete. The selected point will be colored green. Click then on the Remove point button.  
Cross-section Length | The cross-section length determines the width of the cross-section images. This length is set to optimal by default, but can you can modify it by selecting the Custom option and indicating the desired cross-section length in the edit field.  
  
##### X-Ray image (Panoramic option)

A yellow border surrounds the X-ray image or parallel view. Its thickness can be set in the Reslicing section of the Preference Settings (**Edit > Preferences > Reslicing**) from the menu bar. 

To display the x-ray view, click on this button ![](Mimics Reference Guide/../Resources/Images/Along%20Curve/030000014.png), next to the parallel view. To return to the parallel view, click on this button: ![](Mimics Reference Guide/../Resources/Images/Along%20Curve/030000015.png).

![](Mimics Reference Guide/../Resources/Images/Along%20Curve/030000016.png)
