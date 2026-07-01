### Objects

![](Mimics Reference Guide/../Resources/Images/Analysis_PMtab_466x251.png)

#### List of created Objects

Name |  Name of the object. By clicking on the name of the object, it can be renamed.  
---|---  
Visible |  Lists if the object is visible or not  
_Contour Visible_ | Lists if the contour of the Part or STL is visible on the 2D images or not. You can change the visibility of the contours by clicking on the visibility icon.   
Transparency  | Lists the transparency settings of the Part or STL, possible options are: opaque (= not transparent), low, medium and high. Change the transparency by clicking on the icon in the transparency column. To see your objects transparent, the transparency button has to be enabled  
_Images_ | The link to the image set. The link to the image set can be modified via the dropdown list.  
  
#### Functions on Objects

New |  Creates a new Object. The object menu appears :  ![](Mimics Reference Guide/../Resources/Images/Images2/image33.png) Click on the desired object to go to the correct help page.  
---|---  
Delete |  Deletes the selected objects.  
Properties |  Displays the properties of the selected object.  
Duplicate |  Creates a copy of selected objects  
_Filter_ | Filters objects by name and type. As you type, Mimics will start filtering and provide a list of objects that match the text you typed. The list of objects is simultaneously filtered for type of objects and name of the objects.  
  
**Filter objects in Object tab**

Assuming that there are the following objects in the Objects tab:

![](Mimics Reference Guide/../Resources/Images/Objects_tab_examples_446x256.png)

To filter the objects, type a string in the filter area.

Filter by type |  To filter the Parts, type "Part". The result will be the following: ![](Mimics Reference Guide/../Resources/Images/objects_tab_filter_parts_301x159.png) Notice that in case an object of different type contains the string "part" in it's name (in this case a Point that contains the string "part" in it's name) will also appear in the list. You can filter for different types of objects that are shown in the Object tab. Some of them are: point, part, stl, circle, spline etc.  
---|---  
Filter by name |  To filter all the objects that contain the string LV in their name, type "LV". The result will be the following: ![](Mimics Reference Guide/../Resources/Images/objects_tab_filter_by_name_317x166.png) Only the objects that contain the string LV are shown in the list  
Filter a group |  To filter all the objects that are analytical primitives, type "primitive" or "analytical". The result will be: ![](Mimics Reference Guide/../Resources/Images/objects_tab_filter_primitives_327x172.png)  
  
##### Functions on 3D parts. 

Can be accessed by right-clicking on a part.

New |  Creates a new Part. The Calculate Part window appears.  
---|---  
Copy |  Copies the Part to the clipboard. The Part can then be pasted in another project.  
Delete |  Deletes a Part.  
Properties |  Gives the properties of the calculated Part.  
Duplicate |  Duplicates the selected Part.  
Move |  Activates the handles to move the STL to a new positions  
Rotate |  Activates the handles to rotate the STL around its axis  
_Remesh_ | Opens 3-matic with the selected part ready for remeshing  
_Transform_ |  Opens the transformation dialog: ![](Mimics Reference Guide/../Resources/Images/Transformation_dialog.png) In this dialog, you can enter a transformation matrix, invert the transformation matrix if needed and apply it on the selected STL file. You can also load a transformation matrix file, written out by Mimics when reslicing or cropping a project or when doing an STL registration.  
_Action_ | Lists the available functions on the selected Part  
  
##### Properties of an Object

When you click on the Properties button in the Objects list, a dialog box will open. It's form depends on the type of the object chosen:

Part |  ![](Mimics Reference Guide/../Resources/Images/3D_Properties_351x343.png)  
---|---  
Point |  ![](Mimics Reference Guide/../Resources/Images/PMTAB_Point_Properties.png)  
Line |  ![](Mimics Reference Guide/../Resources/Images/PMTAB_Line_Properties.png)  
Circle |  ![](Mimics Reference Guide/../Resources/Images/PMTAB_Circle_properties.png)  
Sphere |  ![](Mimics Reference Guide/../Resources/Images/PMTAB_Sphere_properties.png)  
Plane and Normal plane |  ![](Mimics Reference Guide/../Resources/Images/PMtab_Plane_properties.png)  
Cylinder |  ![](Mimics Reference Guide/../Resources/Images/PMTAb_cylinder_properties.png)  
Spline (Thin Structure) |  ![](Mimics Reference Guide/../Resources/Images/PMTab_spline%20properties_357x252.png)  
  
Mimics displays the name, color and transparency of the object. The volume, surface, outer dimensional parameters and the number of points and/or triangles are also shown, which give a good idea if reducing of the file for further applications is needed.

You can change the transparency of the Part and make objects transparent. Drag the Transparency slider at the bottom of the 3D properties dialog. If the slider is all the way to the right, the 3D surface is opaque and nothing below the surface can be visualized. If the slider is all the way to the left, the 3D surface is completely transparent. In order to see the Part transparent, you need to click the Toggle Transparency button ![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/0200002F_22x22.jpg) in the 3D toolbar.

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/03000030_194x169.png) |  ![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/03000031_192x167.png)  
---|---  
Opaque 3D |  Transparent 3D  
  
If the Details button on the 3D Properties dialog is selected, some measurements of the Part are displayed.

##### Rotate

When you select a Part and click on the Rotate button, rotation handles will appear around the Part. 

You can rotate the Parts around the selected axes of rotation by grabbing one of the colored rotation handles. You can also rotate the object around an axis perpendicular to the camera by grabbing the outer ring of the rotation tool. To change the pivot point, select the yellow box and move it to its new position.

The orientation of the rotation handles can be altered in the rotation dialog. You can choose to rotate along the views, inertia or a user defined axis. The axis can be defined with selection of analytical points that already exist in the Project Management tab or with points that are created with the Measure and Analyze tool.

The user can select to position the pivot point initially at the center of the bounding box, at the mass center or fixed to a user defined point. The defined point can be an analytical point that already exist in the Project Management tab or a point that is created by the Measure and Analyze tool.

![](Mimics Reference Guide/../Resources/Images/3D_Rotate.png)

The rotation value is shown in the status bar.

![](Mimics Reference Guide/../Resources/Images/rotate1.PNG) |  ![](Mimics Reference Guide/../Resources/Images/Capture_280x293.png)  
---|---  
  
To rotate in discrete steps, enter an angle in the offset axis field along which you want to rotate. By clicking Apply you will rotate by the given angle.

##### Move

When you select a Part and click on the Move button, translation arrows appear. To translate the Part grab a translation arrow and move the mouse. To translate the object parallel to the viewing plane, grab and move the middle of the tool. 

You can change the translate axes in the Move dialog. You can choose to translate the object parallel to the viewing axes, inertia or along a user defined axis.

The axis can be defined with selection of analytical points that already exist in the Project Management tab or with points that are created with the Measure and Analyze tool.

![](Mimics Reference Guide/../Resources/Images/3D_Move.png)

The translation value is shown in the status bar.

![](Mimics Reference Guide/../Resources/Images/Move2.PNG) |  ![](Mimics Reference Guide/../Resources/Images/Move%201.PNG)  
---|---  
  
Translating in discrete steps is possible by entering a reposition measure in one of the offsets axes. By clicking on Apply you will translate the object by the selected offset.
