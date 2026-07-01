##### Cut Centerline Ending

The Cut Centerline Ending tool allows you to cut the centerline endings, in order to obtain adequate flat surfaces that can serve as inlet/outlet surfaces for CFD analyses. This tool can be accessed by going to the action list in the Analysis objects tab in the project management.

![](Mimics Reference Guide/../Resources/Images/CenterlineActions.png)

![](Mimics Reference Guide/../Resources/Images/image19.png)

In the Cut Centerline Ending dialog, press the Indicate button to indicate one or more cutting surfaces along the centerline of the blood vessel in the 3D view. The centerline and the STL will be cut according to these cutting surfaces which are perpendicular to the centerline. The original STL is not removed, and a new STL with cu endings is created. The original centerline is replaced by the cut version.

It is only possible to create a cutting surface in a centerline part between one end point and one branching point, and not between two branching points.

The centerline extraction results in a centerline which is central in the blood vessel except near the endings of the blood vessel. In this region, the centerline deviates to the contour of the blood vessel. This Cut Centerline Ending tool allows removal of this incorrect part of the centerline.

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/030001CF_187x197.png) |  ![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/030001D0_184x195.png)  
---|---  
Centerline with irregular endings |  Centerline with flat surfaces after Cut Centerline Ending tool
