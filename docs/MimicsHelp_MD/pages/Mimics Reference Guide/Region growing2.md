### Region growing

The region growing tool makes it possible to split the segmentation created by thresholding into several objects and to remove floating pixels. 

Click the Region grow button ![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/020000BD_25x25.jpg) or press Ctrl + R. The mouse is now cross-shaped and the Region Growing window is on the screen.

Select the Source (= Green) and Target mask (= New Mask). Click the left mouse button on one point in the green area of the object of interest (which is a part of the current segmentation object, i.e. part of the skull). The program starts to calculate the new segmentation, all points in the current segmentation object that are connected to the marked point will be used to form a new mask. The new segmentation is colored yellow.

Click the Close button to close the Region growing window.

To make this new mask active, select "Yellow" in the Visualization toolbar. Clicking on the green glasses will hide the green mask. Clicking the button again will make the green mask visible. 

Check the mask on different images. When we check the images, we see that everything looks fine. It�s time to build a 3D representation.

Note: Thresholding needs to be done before region growing, since all previous work is lost after changing the threshold value.
