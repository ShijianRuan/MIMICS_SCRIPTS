#### Along Plane

The reslice along plane function allows you to reslice the stack of images along a specified plane. As a result, three orthogonal views are obtained with one view parallel to the chosen plane. 

It is possible to:

  * Create plane: by selecting this option the cursor will change to a pencil. Indicating three points on the 2D images or on the Parts will define the plane.
  * Select Plane: Select an already created plane (an Analysis plane or a Normal plane): a window is shown in which all possible planes are listed.
  * Interactive MPR: Interactively reslice the stack of images: after choosing this option, a reslice plane parallel to the current axial view will be created. The crosshair in the Interactive MPR mode will be thicker than the normal crosshair. Rotate the reslice plane by grabbing the crosshair using the left mouse button. For repositioning the reslice plane go to the intersection of the crosshair, grab and move the center of the crosshair using left mouse button. This is possible in all 2D images.


In all cases, the reslice plane (RP) is added to the Reslice Objects tab. When the interactive reslice option is chosen, the last position is saved as reslice plane (RP) in the Reslice Objects tab.

##### Reslice Objects tab 

Several functions on the objects can be accessed via the Reslice Objects tab in Project Management area.

![](Mimics Reference Guide/../Resources/Images/ResliceObjects_OptionsTab.png)

New |  Creates a new Reslice Object. The different options appear :  ![](Mimics Reference Guide/../Resources/Images/AlongPlane_options.png) Click on the desired option.  
---|---  
Delete |  Deletes the selected objects.  
Properties |  Displays the properties of the selected object.  
Duplicate |  Duplicates the selected object  
Actions |  The action button lists all the available actions on the Reslice objects. ![](Mimics Reference Guide/../Resources/Images/ResliceObjects_options.png)  
  
##### Properties

![](Mimics Reference Guide/../Resources/Images/ReslicePlane_Properties.png)

Label | Change the name of the reslice plane  
---|---  
Modify

> ![](Mimics Reference Guide/../Resources/Images/ReslicePlane_PropertiesRotate.png) Rotate ![](Mimics Reference Guide/../Resources/Images/ReslicePlane_PropertiesResize.png) Resize

|  Change the orientation and size of the reslice plane. Change the orientation of reslice plane by moving the rotation handles to the desired position or by manually entering the values in the dialog box. Change the size of the reslice plane by entering the values in the dialog box or by grabbing the mouse cursor ![](Mimics Reference Guide/../Resources/Images/Along%20Plane/03000003.png) and moving to the desired location.  
Reslice step | Defines the distance by which the reslice plane will move through the image cube.  
Help | Opens the respective help chapter from the help file.  
  
Note: When creating a reslice plan by indicating 3 points, these points should not be placed on a straight line.
