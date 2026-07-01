### Calculate Polylines

The calculated polylines are the high resolution segmentation contours for the current segmentation object identical to how they will be calculated by STL+. It can be used to determine the exact threshold value(s) for a correct segmentation. Pressing the Calculate Polylines button results in the following window:

![](Mimics Reference Guide/../Resources/Images/CalculatePolylines.png)

To start calculation, select the desired masks and click Calculate.

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/0200012E_189x266.jpg) ![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/0200012F_174x266.jpg)

If your selection contains a polyline that has a deformed shape, update that polyline in the following way:

  * Select the Edit tool![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/03000130_22x22.png).


  * Alter your segmentation mask with the draw or erase function.


  * Click the Update button ![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/03000131.png) or select Update Polylines from the Segmentation menu.


Repeat this in every slice where the polylines need to be updated. If polylines need to be updated in a lot of slices you better alter first your segmentation mask in every slice and then recalculate the polylines for the whole mask.
