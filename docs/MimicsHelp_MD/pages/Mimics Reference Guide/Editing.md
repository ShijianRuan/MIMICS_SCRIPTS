### Editing

In the axial image, go to position 387,00. Press the Edit masks button ![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/020002C6_25x25.jpg). The edit toolbar is displayed on your screen. Your cursor has become a little square. If not, go to Type and select a square from the drop down list. Notice that the length and the width of the square are displayed and can also be altered. The easiest way to change the size is to press the control key and your left mouse button simultaneously and to move to the right/left to make the square bigger/smaller.

![](Mimics Reference Guide/../Resources/Images/EditMask_obturator_621x77.png)

The three modes available are listed below. To make a mode active, just click in the little circle on the left of the mode or press the first letter of the desired mode. When the edit mode is not yet selected and you use the shortcuts between parentheses below, the edit toolbar appears and the associated mode is activated.

  * Draw (Ctrl + D): Every pixel that lies within the shape of your cursor, while pressing the left mouse button, will get the color of your active mask. In other words, you add pixels to the active mask by going over the pixels with the square.


  * Erase (Ctrl + E): This mode is the opposite of the draw mode. You remove all the pixels from the active mask by moving the square (keeping the left mouse button pressed) over the pixels in the image.


  * Threshold (Ctrl + T): This mode is used to set a local threshold. This means that if you apply a local threshold in a particular area of one image, this threshold doesn�t apply to other images in the project. Remark that the threshold we�ve set in the beginning of this case was global and it applied to every image in the dataset. 


When you activate this mode, a box with the two default threshold values is displayed on your screen. To set a local threshold, press one of the two arrow buttons and double click on a threshold value. After you have changed the value, press Enter. When moving the square over the image while pressing the left mouse button, every pixel that comes to lie within the square and has a threshold within the threshold range you set, will be added to the active mask. On the other hand, all the pixels that were already part of the active mask and that don�t have a gray value within the range will be removed from that mask.

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/030002C8.png)

For the current case we are only going to use the draw and the erase mode. In the Simon case we already illustrated the threshold mode.

Working on the axial image in position 387 activate the Erase mode (Ctrl + E) and set a very large square (for example, 200 by 200). Press your left mouse button and wipe off all the color in the image. Be sure not to forget any pixels! Close the Edit toolbar. 

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/030002C9_211x187.png) ![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/030002CA_206x187.png)

Notice in the sagittal image that one layer is shown in gray values. In the figure below, the sagittal image is displayed and the arrow points to the layer that has been removed from the active mask (the slice indicator is moved down to see this).

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/030002CB_352x224.png)

To see the result of erasing the mask in one layer, we will now perform a region growing ![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/0300029B_22x22.png). Select the axial image at a position lower than 387,00, (= the position of the image we removed from the active mask). Press the Region Growing button, a window will be displayed on the screen.

![](Mimics Reference Guide/../Resources/Images/RegionGrowing_obturator.png)

Check both the Multiple Layer and Leave Original Mask checkboxes and click on an arbitrary position in the active mask. You see that all the images at a position lower than 387,00 are put into a new mask (yellow mask in figure below). Close the region growing toolbar.

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/030002CD_364x232.png)

Why are the images above this position not included into the new mask? As you already know, a region growing looks for pixels that are connected to each other and puts them into a new mask. But, because we have disconnected the lower images from the higher ones, we have limited the area of the region growing. This will be the trick we will use to get our region of interest into a separate mask.

How will we proceed? In the same way as above, we are going to erase a complete layer from the active mask on every side so that our region of interest is completely surrounded by these removed slices. After we perform a region growing within that region, we should have the oral cavity and the surrounding tissue in one mask, like we wanted.

Activate the axial image. Go to position 362,00 and press Ctrl +E (or press the Edit masks button and select the Erase mode). Make a big square and erase all the pixels from the active mask. Take a look at the sagittal image. Two horizontal lines are shown in gray values. The top and the bottom of our box are now defined.

To set the left and right boundaries of the box, you have to remove two layers from the mask in the sagittal image. Try to visualize the situation and make sure you understand why we will now operate in the sagittal image. Erase all pixels from the active mask at position 126,49 (left boundary) and 42,05 (right boundary) in the sagittal image. In the axial image two vertical lines in gray values are visible.

To close our box, a separation still has to be made on the posterior side. Activate the coronal image and remove all pixels from the active mask at position 76,61. In the axial image the removed layer is visible. Setting a boundary on the anterior side is not necessary. In figure 5-10 you can see the boundaries of the mouth cavity on the yellow mask.
