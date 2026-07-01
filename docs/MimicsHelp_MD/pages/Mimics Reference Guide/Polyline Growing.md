#### Polyline Growing  
  
Starts the Polyline Growing mode. The Polyline Selector Toolbar appears:

![](Mimics Reference Guide/../Resources/Images/Analyze_PolylineGrowing.png)

The Polyline Growing tool provides the capacity to create several sets of polylines. The operation can be performed on one single polyline or in 3D.

The parameters needed are:

From |  The set from where the polylines are grown  
---|---  
To |  The set to where the polylines are grown to. This can be an existing selection or a new created one.  
Correlation |  A measure for the strength of matching (%). This is done with a geometrical comparison.  
Auto multi-select |  This parameter can be turned on/off. If it is turned off only the current polyline will be grown. If this parameter is turned on, all polylines according the Matching parameter will be grown.  
Keep Originals |  By default the polyline that is grown into a new set is removed from the source set. You can choose to keep this polyline in the source set by enabling the Keep Originals checkbox.  
  
Selection is done on the images in 2D by drawing a rectangle (Press the left mouse button to indicate a corner of the zoom rectangle, drag and release to indicate the opposite corner) around the desired polyline, or simply by clicking on the desired polyline.

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/020001AE_152x246.jpg) ![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/020001AF_151x243.jpg)

If there is a polyline where the shape is a bit deformed, you can alter it by editing on your segmentation mask. In the edit mode you will be able to update the polylines.
