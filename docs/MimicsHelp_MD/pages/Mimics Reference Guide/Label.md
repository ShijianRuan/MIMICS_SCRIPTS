### Label  
  
![](Mimics Reference Guide/../Resources/Images/Label.png)

Label |  Enter label text here  
---|---  
Mask |  The mask that will contain this label  
Font |  Type of text font. To choose a font, use the Set Font button on the right of the Font field.  
Size |  Text size in mm  
Depth |  Depth of the text in mm. The label will start from the current position and will move upwards, unless the top image is reached, then the label will be copied downwards.  
2D slice-able label |  Determines if the label is 2D slice-able or not. 2D slice-able labels are restricted to horizontal or vertical direction.  
Horizontal or Vertical  |  Orientation of label  
View from top or View from Bottom  |  Orientation of label  
  
Note: Changing the font of labels is not possible with 2D slice-able labels. 

Labels can be exported in STL format in any direction (don't check the 2D slice-able label button). If you want your labels to be visible in exported sliced files, check the 2D slice-able label button. When this option is enabled, the labels are restricted to horizontal and vertical direction.

You can move the label on the images (the mouse cursor changes to a cross). Keep the left mouse button down and place the label in the desired position (this can be done in all views).

To further edit or delete the created label, click the right mouse button on the label. A pane appears with the function Delete, which will delete this label, and the function Properties, which will display the Label Properties Dialog box.

The labels might look incorrect in the 3D View of Mimics. The "View from Top/Bottom" is related to the Top/Bottom of the stack of images.
