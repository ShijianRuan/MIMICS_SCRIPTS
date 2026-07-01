#### Wrap

The wrap function creates a wrapping surface of the selected entities. This tool is particularly useful for medical parts, to filter small inclusions or close small holes. Furthermore, the function is a useful tool towards Finite Element Analysis, where an enveloping surface is needed.

To launch the wrap tool, select the corresponding option from the 3D Tools menu or you can go to the context menu by clicking on this button ![](Mimics Reference Guide/../Resources/Images/ContextMenuIcon.png) in the project management tab and select **Wrap** in the options.

The following window will pop up:

![](Mimics Reference Guide/../Resources/Images/Wrap2_559x168.png)

Functions on Objects to Wrap

Visible |  Lists if the object is visible or not by means of glasses. Click on the glasses to change the visibility of the object.  
---|---  
Contour visible |  Lists if the contour of the object is visible or not by means of glasses. Click on the glasses to change the visibility of the contour of the object.  
Smallest Detail |  Corresponds to the size of the triangles on the newly created surface.  
Closing Distance |  Determines the size of gaps that will be wrapped away via the operation.  
Dilate result | If the checkbox is ticked, the result after wrapping will be dilated such that the pixels around the extremities of the mask are included.  
Protect thin walls |  If this option is not selected, there is no protection of thin walls. Depending on the smallest detail, it is possible that walls with a thickness within the same range are collapsed. If this option is selected, thin walls will be preserved. This means, however, that the resulting model will be slightly thicker than the original, entirely depending on the smallest detail parameter that is chosen.   
Keep originals |  If the keep originals checkbox is checked, the original objects will be kept, otherwise they will be deleted and only the cut objects will remain.
