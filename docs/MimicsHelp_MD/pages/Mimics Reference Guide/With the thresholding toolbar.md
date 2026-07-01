#### With the threshold tool

Set the threshold of the active mask. You can set or change the active mask in the Masks tab of the project management. 

![](Mimics Reference Guide/../Resources/Images/Threshold/Threshold_dialog_box_307x342.png)  
---  
Threshold dialog box  
  
Threshold is used to create a first definition of the segmentation object. The object can be defined based on one lower threshold, or based on a lower and a higher threshold. In the former case, the segmentation object will contain all pixels in the images with a value higher than or equal to the threshold value. In the latter case, the pixel value must be in between both threshold values to be part of the segmentation object. The threshold value can be changed by moving the sliders in the thresholding toolbar with real time visual feedback. The threshold value will be displayed in the threshold toolbar and the segmentation area is changed accordingly. 

With the two sliders a minimal and a maximal threshold can be set. (Mostly only the minimal value needs to be set) In the Min and Max box, a threshold value can be filled in or the value can be increased or decreased using the up-down controls (ideal for fine tuning the threshold).With the shortcut, the sliders of the histogram can be moved to interactively change the threshold values and immediately review the colored pixels on the images. Press the shortcut SHIFT + CTRL on the keyboard and the left mouse button for the Minimum threshold value. While pressing down, move the mouse to change the histogram slider. For the Maximum threshold values do the same but use the right mouse button.

Minimum threshold | Minimum defined threshold of the mask  
---|---  
Maximum threshold | Maximum defined threshold of the mask  
Predefined thresholds sets | The predefined threshold allows you to quickly select a threshold for a specific tissue type.  
The threshold can still be adapted to your needs.  
Fill holes | Fills the holes in the active mask  
Keep largest | Keeps the largest part of the mask if there are several disconnected parts in a mask.  
  
Additionally the region of interest can be selected by adjusting the crop box on the image views.

![](Mimics Reference Guide/../Resources/Images/CT_Heart_Cropbox_621x347.png)

i | The upper and lower threshold limit is limited to the maximum and minimum intensity in the project.  
---|---  
  
To set the threshold of the active mask, press the Apply button.
