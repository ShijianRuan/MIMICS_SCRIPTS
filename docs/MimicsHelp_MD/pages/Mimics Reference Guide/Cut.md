#### Cut  
  
There are four different cutting tools available under Cut option, _Cut with Plane, Cut with Polyplane, Cut with Curve and Cut Orthogonal to Screen._

##### Cut with Plane

To launch �Cut with plane� tool, click �With Plane� button on 3D tools � Cut group. The following dialog appears:

![](Mimics Reference Guide/../Resources/Images/cut_with_plane_3D%20tools.png)

At first, select objects you want to cut and then specify cutting plane. It is possible to select existing plane from the list or draw new one. To draw new cutting plane you need to click Draw button, after that mouse cursor will be changed to pencil and you can proceed drawing a plane on 2D or 3D views.

![](Mimics Reference Guide/../Resources/Images/cut_with_plane_3D%20tools_example_1.png) ![](Mimics Reference Guide/../Resources/Images/cut_with_plane_3D%20tools_example2.png)

Functions on Cut with Plane

Objects to Cut |  List of objects to cut. Click on an item to select, hold Ctrl or Shift pressed to select multiple items to cut with plane. Clicking on hidden item turns item�s visibility ON.  
---|---  
Cutting plane |  Select an existing plane from the list as a cutting plane.  
Draw | Button launches Draw plane functionality.  
Finite / Infinite |  Switch between options to cut with: \- Infinite plane or  \- Finite plane with defined width, height and thickness greater than 0.01 mm.  
Thickness |  If �Finite� option is selected � default value � 0.01 mm. It is not possible to set 0 mm thickness. If �Infinite� option is selected, �Thickness� field is hidden, since it is not possible to cut with infinite plane with non-zero thickness   
Preview | Preview result of cutting  
Keep originals | If checked the input parts will be kept, otherwise they will be deleted and only the result objects will remain.  
Split results | If checked the cut result will be saved as separate (splitted) parts.  
  
To adjust the size of the finite cutting plane in case it does not fully intersect with selected object launch the plane�s properties dialog window and adjust the size of the plane.

![](Mimics Reference Guide/../Resources/Images/adjust_cutting_plane_size.png)

##### Cut with Polyplane

To launch �Cut with polyplane� tool, click �With PolyPlane� button on 3D tools - Cut group. The following dialog appears and mouse cursor is changed into a pencil ready to indicate cutting path when hovering any viewport.

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/03000241.png)

With this pencil you can draw a cutting path on 3D or 2D viewports. To add next point to cutting path - click on viewport, to finish cutting path creation � you should double click on viewport. After drawing, you can adjust the properties of the cutting path by clicking on the Properties button.

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/02000242_259x264.jpg)

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/02000243_229x264.jpg)

The cutting path will be visible in 3D and in 2D (if the option "Contour Visible" is selected)

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/02000244_277x264.jpg)

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/02000245_247x265.jpg)

Functions on Objects to Cut

Visible |  Lists if the object is visible or not by means of glasses. Click on the glasses to change the visibility of the object.  
---|---  
Contour visible |  Lists if the contour of the object is visible or not by means of glasses. Click on the glasses to change the visibility of the contour of the object.  
![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/02000246_25x24.jpg)Show only selected objects |  Makes the selected objects visible and the unselected objects invisible.  
Keep originals |  If the keep originals checkbox is checked, the original objects will be kept, otherwise they will be deleted and only the cut objects will remain.  
_Split Result_ | If checked the cut result will be saved as separate (splitted) parts  
  
Functions on Cutting Paths

New |  Allows you to create a new cutting path.  
---|---  
Properties |  Displays the properties of the selected cutting path:  ![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/03000229_313x243.png) In this properties dialog you can change the depth, thickness and the extensions at the front and the end of the cutting path. You can also define if the cutting path should be closed or not.  
The Preview button can be used to preview the adjusted cutting path before applying the changes.  
Visible |  Lists if the cutting path is visible or not by means of glasses. Click on the glasses to change the visibility of the cutting path.  
Contour Visible |  Lists if the contour of the cutting path is visible or not by means of glasses. Click on the glasses to change the visibility of the contour of the cutting path.  
  
Cutting paths can be adjust after their creation by left-clicking on the points of the cutting path and dragging them. You can also change the angle of the cutting path by left-clicking and dragging the red arrow on the cutting path.

##### Cut with Curve

By defining a contour on the Part you can make more complex cuts. 

To launch �Cut with Curve� tool, click �With Curve� button on 3D tools - Cut group. The following dialog appears and mouse cursor is changed into a pencil ready to indicate cutting line when hovering 3D viewport.

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/03000247.png)

Indicate points all around the Part to indicate your cutting path. When you indicate the points, a red line will appear on the 3D that represents your cutting line. You finish the cutting path by double clicking the left mouse button (or click once the right mouse button). A yellow line will appear which represents the extension of the cutting path. This extension can be adjusted in the main dialog box. Make sure that the extensions aren�t crossing the Part. The cut will only complete when all extensions are floating above the Part. 

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/03000248_143x125.png) |  ![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/03000249_146x127.png) |  ![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/0300024A_150x135.png)  
---|---|---  
  
You can still adjust the control points. Select a point by holding the left mouse button and drag to a new location. 

After you�ve checked the extensions click OK, the object is now cut AND split at the same time.

Functions on Objects to Cut

Visible |  Lists if the object is visible or not by means of glasses. Click on the glasses to change the visibility of the object.  
---|---  
Contour visible |  Lists if the contour of the object is visible or not by means of glasses. Click on the glasses to change the visibility of the contour of the object.  
![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/0300024B_25x24.png)Show only selected objects |  Makes the selected objects visible and the unselected objects invisible.  
Keep originals |  If the keep originals checkbox is checked, the original objects will be kept, otherwise they will be deleted and only the cut objects will remain.  
Split Result | If checked the cut result will be saved as separate (splitted) parts.  
  
Functions on Cutting Lines

Indicate |  Shows the indicate tool and enables you to indicate a cutting line.  
---|---  
Close |  Closes the cutting line.  
Delete Last |  Deletes the point of the cutting line that was drawn last.  
Extensions |  Sets the distance between the cutting line and its extension  
Thickness | Set thickness of the cutting line.  
  
##### Cut Orthogonal to Screen

To launch �Cut orthogonal to screen� tool, click �Orthogonal to Screen� option on the 3D tools - Cut group. The following dialog appears and your cursor will change.

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/0300024C.png)

You have to select an object to cut from the list before you can preview or apply the cut.

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/0200024D_448x389.jpg)

Functions on Objects to Cut

Visible |  Lists if the object is visible or not by means of glasses. Click on the glasses to change the visibility of the object.  
---|---  
Contour visible |  Lists if the contour of the object is visible or not by means of glasses. Click on the glasses to change the visibility of the contour of the object.  
![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/0300024B_25x24.png)Show only selected objects |  Makes the selected objects visible and the unselected objects invisible.  
Keep originals |  If the keep originals checkbox is checked, the original objects will be kept, otherwise they will be deleted and only the cut objects will remain.  
  
Functions on Cutting Lines

Indicate |  Shows the indicate tool and enables you to indicate a cutting line.  
---|---  
Close |  Closes the cutting line.  
Delete Last |  Deletes the point of the cutting line that was drawn last.  
_Clear All_ | Delete a cutting line.
