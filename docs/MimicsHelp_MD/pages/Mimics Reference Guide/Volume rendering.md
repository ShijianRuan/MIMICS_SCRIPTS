#### Volume rendering  
  
Volume rendering ![](Mimics Reference Guide/../Resources/Images/Volume_rendering.png) allows you to quickly visualize your 2D image data as a Part without any segmentation. The Part is build up out of the voxels representing the dataset. The transparency of the voxels is determined based on their gray value. Volume rendering is a pure visualization tool and cannot be used for anything else (e.g. exporting).

You can find the interface for the volume rendering in the Bottom Panel area:

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/0300003F.png)

Defining the opacity

This interface shows a histogram which represents the gray values or Hounsfield units of the dataset. The transparency of the gray values is set by the opacity lines on the histogram. The higher a line is positioned the more opaque the voxels within that range will be visualized. 

In the first column the line representing the low Hounsfield values is positioned to the bottom. Subsequently the voxels are represented transparent. In the second column the line is positioned higher which makes the voxels opaque.

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/03000040_214x187.png) |  ![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/03000041_214x187.png)  
---|---  
![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/03000042_214x191.png) |  ![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/03000043_214x191.png)  
  
Move a point |  ![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/03000044.png) |  Position a point click left and drag the point to its new position  
---|---|---  
Add a point |  ![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/03000045.png) |  Click left on a line to add an extra point  
Delete a point |  |  Right click on a point and select delete to delete a point  
  
Defining the color

The graph below the histogram defines the color of the rendered voxels. You can choose a color for each point in the graph by right-clicking on the points and choosing Change Color. Mimics will then create an interpolated shading between the different control points. The same system for adding, moving and deleting bullets is available as for the opacity line.

Predefined settings

On the bottom of the volume rendering interface you find a dropdown list with predefined settings. The predefined settings are optimized for CT image and allow you to quickly select bone, soft tissue or both

You can save your current setting as a predefined setting by clicking on the save button. To delete your predefined setting, select it from the dropdown box and click on the delete button.
