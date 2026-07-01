# Fluoroscopy Simulation

The Fluoroscopy simulation tool generates simulated fluoroscopic images from the image data based on the view angle position. In practice, the fluoroscopic images are obtained with the help of a C-arm.

**Fluoroscopy simulation viewport**

![](Mimics Reference Guide/../Resources/Images/FluoroscopySimulation_Layout_621x347.png)

In the fluoroscopy simulation viewport the 3D view is replaced by the simulation view which displays the simulated fluoroscopy image along with the projected 3D models, if any. There are two modes available for selecting the resolution of the simulated image. The default setting is on Low resolution.

  * Low (resolution 250 x 250)
  * Hi (resolution 600 x 600)


The control panel in the fluoroscopy simulation mode shows a list of the 3D objects that can be simulated. Only the objects that are linked to the active images or objects that are not linked to any images are shown in this list. These 3D models can be used for simulating the segmented anatomy.

To simulate the model, tick the check box in front of the desired 3D model(s). The image will be updated with the selected model(s). The contrast of the projected 3D model can be changed by manually entering the values in the **Contrast** column of the list.

![](Mimics Reference Guide/../Resources/Images/SaveScreenshot.png) | Saves the active simulated fluoroscopic image in a .png file format to the defined location. This button is present on the right side of the simulation viewport.  
---|---
