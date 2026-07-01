#### The 3D LiveWire Interface

![](Mimics Reference Guide/../Resources/Images/3DLiveWire.png)

Target |  The mask that will be created when applying the 3D LiveWire tool  
---|---  
Automatic contour |  View for which the automatic contour will be generated, based on the points selected on the boundaries of the object in the two other views, and that will lead to the calculation of the mask.  
Apply parameters to all contours |  If this option is ON, the parameters selected for Gradient magnitude and Attraction will be applied to all the automatically calculated contours. If this option is OFF, the parameters will be applied only for the automatically calculated contour present in the slice displayed in the screen.  
Gradient magnitude |  This parameter indicates to which kind of gradients the contour will be attracted. If the selected value is close to 0%, the contour will be attracted to darker regions on the boundaries of the object. If the selected value is close to 100%, the contour will be attracted to brighter regions lying at the boundary of the object/  
Attraction |  The attraction coefficient indicates if some cavities in the boundaries of the object should be taken into account or should be neglected. If a value near to -3 is selected, all the small inclusions will be included in the mask. If a value near to 3 is selected, the inclusions will be excluded from the automatic contour.  
Apply parameters to range |  This option allows applying the selected parameters to a range of slices in the dataset. When you click on the Start button, you indicate the first slice of the range. Scroll then through the dataset, until the last slice to which the parameters should be applied and click on the Stop button.  
Segment |  When you click on the Segment  button, a mask is calculated based on the automatic contour.
