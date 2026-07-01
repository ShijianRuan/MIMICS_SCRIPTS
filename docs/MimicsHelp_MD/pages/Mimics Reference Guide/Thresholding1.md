### Thresholding

Thresholding means that the segmentation object (visualized by a colored mask) will contain only those pixels of the image with a value higher than or equal to the threshold value. Sometimes an upper and lower threshold is needed; the segmentation mask contains all pixels between these two values.

For example:

A low threshold value makes it possible to select the Soft tissue of the scanned patient. With a high threshold, only the very dense parts remain selected. Using both an upper and a lower threshold is needed when the nerve channel needs to be selected. Defining a good threshold value also depends on the purpose of the model. If you just want a nice looking model, a lower threshold value is recommended since it will result in a model with fewer holes. On the other hand, when the model serves for modeling prostheses a higher threshold value is preferred.

  * Click the Threshold button ![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/020000BC_25x25.jpg):


To change the threshold value, press the left mouse button on a slider in the Threshold Toolbar and move the slider by moving the mouse (while still holding the left mouse button). 

Some tips for selecting an adequate threshold value: 

Look at different images. You can change images of any view by: 

using the arrow keys, the page up and page down keys 

using the slider on the right in the window border

moving the slice indicators

  * Click the Draw Profile Line button ![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/02000280_25x25.jpg):


In the axial view draw a line over the bone as shown below. To draw this line, click the left mouse button in the soft tissue to indicate the starting point, move the mouse over the bone click. Along this line an intensity profile is generated. The straight horizontal lines represent your current threshold value. Click on Start Thresholding and drag the lower straight-line up/down to set a good threshold. If you want a good visualization model, select a threshold slightly above the intensity plateau of the soft tissue. If your model will serve for modeling prostheses, place the line between the soft tissue plateau and the top value of the bone. If a proper threshold is set, click on End Thresholding to save the current value.

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/03000281_183x204.png) |  ![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/03000282_262x181.png)  
---|---  
  
Zoom in on a part you�re interested in. First, from the pull-down menu next to the zoom button, select Box. Click the Zoom button ![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/02000283_25x25.jpg): the mouse is displayed as a loupe. Click the left mouse button on the image and drag for creating a zoom rectangle, release for zooming. To return to the whole image, click the Fit to screen button ![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/02000284_25x25.jpg). 

A good threshold value for Mimi is about 270 (Hounsfield scale). The threshold value is displayed in the Min. box of the Threshold toolbar. To end thresholding, click the Apply button.

After the thresholding operation a green mask will be created. In a project you can have different masks but you can use the segmentation tools only on the active mask. To choose the active mask, select it in the mask tab in the project management. In case the project management isn�t active, select the project management button ![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/02000285_25x25.jpg) in the main toolbar.

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/03000286_236x217.png)

You can also hide any mask by clicking on the eye of the corresponding color.
