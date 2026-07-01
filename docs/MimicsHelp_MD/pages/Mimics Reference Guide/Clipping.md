#### Clipping  
  
Clipping allows you to visualize the section in which you are interested. It can be used, to evaluate the gray values on the section boundaries or to look inside the model to get a better comprehension of the geometry. The section can be made along the different planes, axial, coronal and sagittal. Several clipping planes can be activated at the same time enabling you to isolate the part of interest.

To enable clipping click on the Enable/Disable Clipping button ![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/02000032_22x22.jpg). The settings of the clipping can be changed in the clipping tab:

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/03000033.png)

Active

By default one axial clipping plane is active. To make more planes active, make sure to check the active icon ![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/02000034.jpg). 

[![](Mimics Reference Guide/../Resources/Images/Clipping1_200x146.png)](<file:///C:/Users/ysermeus/Desktop/refguides/mimics refguide screenshots/Clipping1.PNG>) |  ![](Mimics Reference Guide/../Resources/Images/Clipping2_200x146.png) |  ![](Mimics Reference Guide/../Resources/Images/clipping3_200x146.png)  
---|---|---  
Clipping in the Axial Plane |  Clipping in the Sagittal Plane |  Combined Axial and Sagittal Clipping Planes  
  
The position of the clipped plane in the 3D view corresponds with the position of the active axial, sagittal or coronal image. Scrolling through the 2D images updates the clipped plane in 3D. Also navigation on the clipped 3D is possible by clicking on visible parts on the 3D. When you rotate the 3D along the clipped plane, the visibility automatically reverses.

Type

As stated above you can select multiple clipping planes. For each view you can define two clipping planes. For each of the clipping planes, you can choose which objects need to be clipped. You can choose this by selecting the clipping plane in the list and changing the selections in the dropdown menu ![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/03000038.png). All objects are selected by default.

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/03000039.png) |  ![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/0300003A_193x187.png)  
---|---  
Clipping along two axial and a sagittal plane |  Clipping along the axial and sagittal plane  
  
Clip

By default the direction of the clipping plane is defined by the viewing angle. The direction of the clipping plane can be locked by selecting a clip direction. Click on the clip icon ![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/0200003B.jpg) to lock to a clipping direction or to unlock.

Lock

The location of the clipping plane is locked to the slice position. To unlock the clipping plane from the slice position, disable the lock icon ![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/02000034.jpg). When the lock icon is disabled, you can determine the location of the clipping plane with the slider at the bottom of the clipping tab.

Texturing

You can also choose between three texturing methods:

![](Mimics Reference Guide/../Resources/Images/Texturing_186x166.png) |  ![](Mimics Reference Guide/../Resources/Images/Texturing2_190x160.png) |  ![](Mimics Reference Guide/../Resources/Images/Capture3_231x168.png)  
---|---|---  
No texturing |  Object texturing |  Full slice texturing  
  
When choosing No texturing, only the 3D Object is clipped and you can see inside the 3D Object. When choosing Object texturing, a texture corresponding with the 2D slice is placed in the contours of the 3D Objects. When choosing Slice texturing, the 3D Object is not clipped, but the whole 2D slice is visible in the 3D window.

Note: This technique is only available in OpenGL and direct3D rendering (not in software). To check your rendering option, go to Edit > Preferences and select the OpenGL or Direct3D rendering option in the 3D settings. To accelerate the performance of clipping, activate the hardware acceleration in the same dialog. 
