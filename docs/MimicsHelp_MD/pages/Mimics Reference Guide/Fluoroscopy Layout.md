# Fluoroscopy Layout

**Image and 3D viewport**

The fluoroscopy layout consists of the images and the 3D viewport with overlaying control buttons. The left half of the screen consists of the three standard image views, namely Coronal (Top), Axial (Bottom Left), and Sagittal (Bottom Right). The right half is the 3D viewport with control panel.

All the views consist of an orientation marker![](Mimics Reference Guide/../Resources/Images/OrientationMarker.png)located in the top right corner. This marker updates accordingly to the specified view angles. The respective angle positions are shown on the 3D viewport (bottom right).

**Note:** : The currently active images are always shown in the views. You can modify the active images while the fluoroscopy tool is open.

![](Mimics Reference Guide/../Resources/Images/FluoroscopyView_Layout_621x347.png)

The data is resliced to the specific view angle as entered by the user. It is possible to manipulate the angles in the following ways:

  * Using the sliders on the 3D viewport
  * Using the crosshair on the images
  * Manually entering the values in the boxes beside sliders


The angles are updated as soon as one of the methods is used for manipulation, the images and the 3D view are updated accordingly for the specified angle.

The 3D viewport also has an option to reset the view back to the AP plane ![](Mimics Reference Guide/../Resources/Images/ResetView.png) i.e. RAO/LAO 0� and CRAN/CAUD 0�.

**Control Panel**

Active view | Shows the list of all the fluoroscopy views that have been created. This list is also displayed in the Project Management tab under the Reslice Objects.  
---|---  
AP0 Plane | Defining the anteroposterior (AP) plane allows correcting for differences between the AP position of the CT scan, and the AP position of the fluoroscope. By default, this AP plane is defined normal to the AP position of the image data.  
Visualization | The visualization mode comprises of two modes: 3D View and fluoroscopy Simulation. The 3D view mode displays the 3D model(s) will be displayed. Simulation mode is used for fluoroscopy simulation.  
View alignment |  Free - The Part will not be linked with the rotation of the reslice planes. Reslice Plane - The view will be aligned to the resliced coronal view. As a result, the Part will rotate when manipulating the sliders.  
List of view angles | The list shows all the saved view angles. When a particular view angle is selected from the list, the images are aligned to the respective view angle.  
  
![](Mimics Reference Guide/../Resources/Images/ToggleSliders.png) | Show/hide sliders on the viewport.  
---|---  
![](Mimics Reference Guide/../Resources/Images/CropView.png) | Show/hide the crop box.  
![](Mimics Reference Guide/../Resources/Images/Properties.png) |  The settings for fluoroscopy simulation contain the following options.

  * Name - Name of the fluoroscopy view
  * Distance source to detector - Distance between the X-ray source and the detector
  * Distance source to patient - Distance between the X-ray source and the patient
  * Attenuation coefficient - Attenuation coefficient of the X-ray beams used for simulating the fluoroscopy view
  * Normalize contrast - Artificial normalization of the grayscale to improve the quality of the simulated image in case when too light or dark

  
![](Mimics Reference Guide/../Resources/Images/SaveFluoroscopyView.png) | Save view angles.  
![](Mimics Reference Guide/../Resources/Images/ExportViewAngles.png) | Export view angles.  
![](Mimics Reference Guide/../Resources/Images/DeleteViewAngle.png) | Delete view angles.  
  
**Hint** : The layouts can be changed using the shortcut keys when using Fluoroscopy tool, e.g. press F2 key to switch to image layout.
