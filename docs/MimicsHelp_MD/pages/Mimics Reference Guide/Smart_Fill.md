### Smart Fill

The Smart Fill tool fills holes and cavities of the selected mask by global and local filling. Click on the Smart Fill icon in the Segment menu to open the following dialog box:

![](Mimics Reference Guide/../Resources/Images/SmartFill_window.png)

**Selection**

In the selection field, choose the mask for which holes and cavities need to be filled. To change the selection, choose another mask from the list.

**Global**

The global filling will look at the entire selected mask and try to close the holes and cavities that have a certain size which is defined by the 'Hole Closing Distance' parameter. To define this parameter the closing size of the hole, measured in voxels, needs to be estimated at the border of the mask where the hole leaks to the outside. When pressing on the 'Fill Holes' button the cavities and holes in the selected mask will be filled taking into account the defined parameter for the 'Hole Closing Distance'. 

Areas closed by theSmart Fill tool will be visualized in a green preview color. Via the visibility icon in the dialog box the preview can be anytime hidden to view the input or original images. 

  
Note:   
| A large 'Hole Closing Distance' parameter can cause smoothening of the selected mask at the borders.   
---|---  
For optimal result the selected mask should be separated from surrounding anatomies before filling.  
  
**Local**

With the Local fill it is possible to fill locally holes of the selected mask with the 'Mark Hole' brush. 

_Mark Hole_

The 'Mark Hole' option is a cursor with two circles: a solid inner circle and a dotted outer circle. 

![](Mimics Reference Guide/../Resources/Images/Smart_fill_mark_brush.png)

The inner circle will add the exact colored pixels to the mask and the outer circle will only add the pixels that are needed to close the hole after which it will be filled. 

|  ![](Mimics Reference Guide/../Resources/Images/smart_fill_mark_brush_ex1.png) |  ![](Mimics Reference Guide/../Resources/Images/smart_fill_mark_brush_ex2.png) |  ![](Mimics Reference Guide/../Resources/Images/smart_fill_mark_brush_ex3_431x359.png)  
---|---|---|---  
  
_Erase_

With the Erase tool pixels can be removed from the filled areas in the 2D images. 

Note: To make a selection through the slices, hold the left mouse button and scroll through the slices by either using the mouse scroll wheel or the Up and Down arrow keys.

_Brush parameters_

Diameter |  Defines the diameter of the brush  
---|---
