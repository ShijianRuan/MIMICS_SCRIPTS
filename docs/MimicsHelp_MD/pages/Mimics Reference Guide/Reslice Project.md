### Reslice Images  
  
You can reslice a project by going to the Image menu and then choose the Reslice Images button. Following window will appear:

![](Mimics Reference Guide/../Resources/Images/ResliceProject.png)

Image sets can be resliced according to a straight line in any direction. This will create a new Mimics project on your hard drive. There is an easy-to-use interface available. 

The reslice line can easily be drawn in the 2D or 3D views by clicking on the Draw Line button. The mouse icon will change to a pen and you will be able to draw a straight line in any of the 2D views by clicking once with the left mouse button to indicate the starting point. To finish drawing the line, click again with your left mouse button on the ending point of the line. 

The bounding box of the new image volume will be shown on the 2D images and in the 3D view.

![](Mimics Reference Guide/../Resources/Images/reslice2_338x175.png) ![](Mimics Reference Guide/../Resources/Images/reslice1_336x234.png)

The coordinates of the beginning and end point of the reslice line can be adjusted in the edit fields. You can also rotate the image set that will be resliced around the reslice line by adjusting the rotation angle. Several predefined orientations can be chosen in the orientation dropdown.

Image width, height, pixel size and slice distance can be specified and you immediately get information about the number of slices that will be in the resliced image set. After drawing, you can still adjust the end points of the reslice line. 

When reslicing an image set, Mimics will also write out the transformation matrix that was applied during the reslicing. The file with the transformation matrix will be saved in the folder of the original project.

Upon reslicing a project, a checkbox �**Update CS** � is available. By default, this setting is turned off. When the checkbox is ticked ON, the origin of the coordinate system will be updated to the upper left corner of the first image in the stack, with the axes along the main directions of the image stack. In addition, the transformation matrix used for the update of the CS is saved in .txt format next to the project and can be used to transform imported objects accordingly.
