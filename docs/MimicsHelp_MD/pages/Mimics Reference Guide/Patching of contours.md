### Patching of contours

Since we are only interested in the outer contours, we need to select these out and grow them to a new set of polylines.

Go to layer -523 and zoom in on the right femur in the 2D image (xy plane).

Click the Polyline Growing button ![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/020002B3_25x25.jpg) in the Analyze toolbar.

Set all parameters as displayed in the image below: i.e. the set to start from, the set that will contain the grown polylines. In order to select a polyline, you need to draw a rectangle over it or simply click on its contour. Hold the left mouse button down, drag it and then release the left mouse button.

![](Mimics Reference Guide/../Resources/Images/Polyline_Growing.png)  
---  
![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/030002B5_180x155.png) |  ![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/030002B6_179x154.png)  
  
The growing of the polylines stopped at layer -513 because of a small extension on the bone This needs to be removed in layers �513 and -511. Afterwards, the polylines need to be updated and then we can proceed with the polyline growing: 

  * Click the Edit masks button and go to the Erase mode or press Ctrl + E 


  * Make sure that the Yellow mask is Active


  * Erase the extension on the bone![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/030002B7.png)


  * Press the Ctrl + U key or the Update Polylines button in the Edit toolbar


  * Repeat this for the following images.


Scroll back to image -513 and click the Grow Polylines button. Set "selection 2" as the target polyline and use 96 % as matching parameter. Select the polyline.

Scroll to image �485 (figure below).

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/030002B8_448x384.png)

Image �485 of the Hip

At this slice you see a cavity in the contour. If you want to restore this with editing, keep in mind that it will be the yellow contours that will be updated, so we need to remove the pink polyline first.

Do a Polyline Growing from Selection 1 to a New Set ; be sure to turn Auto Multi-Select off. You can delete this set by selecting it in the Project management and then pressing the Delete button.

Lose the cavity by drawing in the mask and updating the polyline (Ctrl + U).

Similar editing and updating of the polylines needs to be done on slices: -483 till -479, -475, -471 (on the femur head). Don't forget to update for every image.

When all corrections have been made, the polyline growing can continue.

Go back to layer -485 and perform the Polyline Growing (from Set 1 to Selection 2, matching parameter 95 %, Auto Multi-Select on)

Once all the editing is performed properly, all layers until -477 will be stored in Selection 2.

The femur head and the greater trochanter will be grown into new selection sets. The end result should look like the figure below.

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/030002B9_273x278.png)

Polyline sets
