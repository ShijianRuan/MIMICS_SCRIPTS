# Grab

![](Mimics Reference Guide/../Resources/Images/ContourEditing_options.png)

This option allows dragging the contour of the 3D model on all the three image views. When this option is selected, the region of influence (red line) is shown on the contour. Only the part within this region of influence is altered when editing the contour.

**Brush Properties**

Diameter | defines the region of influence on the active slice  
---|---  
Slice Depth | defines the region of influence across slices  
Cone Factor | defines the smoothness of the cone in the edited contour  
![](Mimics Reference Guide/../Resources/Images/Link_Parameters_10x29.png) Link parameters | makes the diameter and slice depth parameters identical  
  
**Note** : The **Grab** tool of the Contour Editing toolbox is intended to fine-tune your Parts by making small edits to the contours visualized in the 2D images. Results are optimal for small changes. For large changes under specific conditions, contours could warp. The effect of the change is immediately visualized, so should a change be undesired, it is advised to simply undo the change and make the edit with smaller alterations in succession.
