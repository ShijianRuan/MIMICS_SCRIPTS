#### Splines  
  
##### Draw Spline

To create splines click on the icon Spline ![](Mimics Reference Guide/../Resources/Images/spline_ribbons.png) in the Analyze menu. When Draw Curve is selected the cursor changes to ![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/030001C5_27x28.png)and the spline toolbox will appear on the screen.

##### Spline toolbox

![](Mimics Reference Guide/../Resources/Images/Analyze_SplineToolbox.png)

  1. Select spline


This function allows you to select the spline you want to edit. The points of a selected spline are in white. The points of a non-selected spline have the same color as the lines of the spline. The action you do is always performed on the selected spline.

  2. Create spline


This function allows you to draw the spline. These are the steps to perform:

  * Indicate the path of the spline with the left mouse button; for every change in direction along the spline, you need to click once. In order to terminate the spline, double click the left mouse button or click the right mouse button.


  * While drawing the spline, you can scroll through the images with the cursor keys if needed.


  * During spline creation, fine adjustments can be made by dragging the points in all 2D images, and in the 3D view, to the desired position. The spline can be moved entirely by dragging the orange line.


  3. Delete a spline


Select the spline you want to delete and click on the Delete spline button.

  4. Add point to spline


Click on the Add point to spline button, hover the mouse over the spline segment were you want to add the point. The cursor will change into a pencil. Click your left mouse button to add a point.

  5. Remove point from spline


Click on the point of the spline you want to delete. The selected point will be colored green. Click on the Remove point from spline button to delete the point.

  6. Close spline


Click on this button to close the spline. 

  7. Spline properties


This button launches the Spline Properties dialog. 

##### Spline Properties

![](Mimics Reference Guide/../Resources/Images/Analyze_SplineProperties.png)

You can view the properties of a spline by clicking on the Properties button in the Project Management, Analysis Objects tab.

Here you can change the color, the diameter and the name of the spline. You can also find some information about the order, length and the number of control points of the curve.

Remark: it is possible to create a spline while using the Interactive MPR mode:

  1. Choose View > Reslice > Along Plane > Interactive MPR, and call the spline tool. When pressing the control key it is possible to reslice the stack of images as to obtain the desired views for correct placing of the spline points.
  2. Choose View > Reslice > Along Curve > Create curve. With this tool, it is possible to create a spline while using the interactive MPR by pressing the control key. The advantage of this second method, compared to the first, is the fact that the blue crosshair of the interactive MPR automatically navigates to the last created control point of the spline.


**Spline compatibility with 3-matic**

Starting from Mimics 18.0, it is possible to copy a spline from a Mimics project and paste it into a 3-matic project as a curve e.g. by using the keyboard shortcuts Ctrl+C and Ctrl+V. The copy-paste action is unidirectional i.e. the spline can only be copied in Mimics and pasted as a curve in 3-matic and not vice versa. For all other objects, the copy-paste behavior remains the same.

The default unit used in 3-matic is millimeter (mm). When copying objects from a Mimics project having different units other than mm and pasting it in 3-matic the following message will appear.

![](Mimics Reference Guide/../Resources/Images/um-mm_WarningMessage.png)

**Note** : When copy-pasting objects with different units in 3-matic, the objects will not be rescaled automatically but can be rescaled manually.
