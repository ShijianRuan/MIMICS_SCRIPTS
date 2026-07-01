### Cineloop  
  
Cineloop is a tool developed mainly for cardiac applications. The heart is a moving tissue and images can be produced for the different phases of the cardiac cycle. The tool allows to visualize the images of each phase in one loop and simulate the heart cycle in the 2D views. Access the tool via the **View** menu -> **Cineloop**![](Mimics Reference Guide/../Resources/Images/cineloop%20icon.png). When the tool is active, the loop is playing in the 2D views. A control panel is superimposed in the 2D views.

Assuming that there are image sets from a single patient and a single study loaded in a Mimics project as they are shown in the image below:

![](Mimics Reference Guide/../Resources/Images/cineloopdatasets_483x152.png)

When the cineloop tool is activated only the image sets that fulfill the following conditions will be included in the loop:

  1. Image sets that have the same geometrical characteristics with the currently active image set by the time that the tool was activated
  2. Image sets that belong to the same patient as the currently active image set by the time that the tool was activated
  3. Image sets the belong to the same study as the currently active image set by the time that the tool was activated


Assuming that the slice of Image 1 (marked in red) is shown in the Axial 2D view, then the slices marked in red from the other datasets that fulfill the requirements above, will be shown in the Axial 2D view while the cineloop is in play mode (see image below).

![](Mimics Reference Guide/../Resources/Images/cineloopdatasetsthatplay_519x159.png)

##### Control panel

. ![](Mimics Reference Guide/../Resources/Images/cineloop%20controlpanel.png)

(1) Currently Active Image set, (2) Speed bar, (3) Previous image set, (4) Play/Pause mode, (5) Next image set, (6) Loop ON/OFF, (7) Close tool. 

Currently Active Image set |  When the tool is in pause mode, the name of the currently shown image set is shown. This is the currently active image set  
---|---  
Speed bar |  Controls the speed of the playback  
Previous Image |  Navigates to the previous image set when the tool is in pause mode. That image set will become the currently active image  
Play/Pause mode |  Plays or pauses the loop. The image set that is shown while the video is in pause mode becomes the currently active one.  
Next Image |  Navigates to the next image set when the tool is in pause mode. That image set will become the currently active one  
Loop ON/OFF |  Turns the loop On or OFF. When the loop is ON the tool plays the loop of the image sets from the beginning to the end continuously.  
Close tool |  Closes the cineloop tool. The last image set that is shown before closing the tool becomes the active one.  
  
##### Interaction with other tools

While the cineloop is in play mode, most of the tools are deactivated. Consequently no actions as Segmentation or Measurements can be performed. Exceptions are the Pan and Zoom operations, the Reslice Along Plane and the Interactive MPR.

While the cineloop is in pause mode, it is allowed to work on the 2D images. All the objects that are created in pause mode follow the normal behavior of Mimics objects as described in the different sections of the reference guide.
