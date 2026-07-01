#### Measure and Analyse

To launch the Measure and Analyze tool, select the corresponding option from the Simulation menu.

The following window will pop up:

![](Mimics Reference Guide/../Resources/Images/maesure%20and%20analysre_178x451.png)

##### Analysis Overview

When you click on the Overview button in the Anthropometric Analysis pane, the following dialog will emerge.

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/0300022F_305x319.png)

##### Creating a new template

To create a new template, simply click on �New� in the overview dialog. A window will pop up allowing you to enter the desired analysis name in the �Analysis� field. 

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/03000230_232x495.png)

Points

When you click on the New button of the Points section, a second field in the pop-up window allows you to define landmarks for the analysis. New landmarks can be created, copied, edited or deleted. Each landmark can have some default properties that can be set when creating the landmark or by editing an existing landmark using the �Edit� function. The properties that can be set are the landmark name, its color and a description. The landmark name can only be set at creation time. If you wish to import existing landmarks, refer to the relevant section in this manual.

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/03000231_339x167.png)

Planes

By clicking on the New button in the Planes section, you can add planes to the template. To define a plane you must first have defined points, or -alternatively- you can define a plane based on other planes you have already created in the template.

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/03000232_448x283.png)

Measurements

In the last field, measurements can be added to the template. Distances, angles and volumes can be measured. For distance either the distance between two points or the distance between a point and a plane can be measured. As for angle, this can be measured using 3 points or using 2 lines (defined by 2 points each). A volume can be measured by selecting at least 4 points. A convex polygon will be created with those points and the volume will be measured. Note that measurements can only be done using points or planes that have already been defined in the template.

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/03000233_448x352.png)

##### Duplicating an existing template

If you wish to create a copy of an existing template, simply click in the �Active� column behind the template you wish to duplicate. A green flag will appear. 

Now click on �Copy� to create the copy. A new window will pop up allowing you to define the template in a similar way as explained under creating a new template.

##### Editing an existing template

To change the properties of an existing template, activate the template by clicking in the �Active� column behind the template. A green flag will appear. Click on �Change� and edit the template as desired.

Note: You cannot change the default templates that are installed with the Simulation module. If you wish to edit the properties of one of the default templates you need to first create a copy of the template and then edit the copy.

##### Removing an existing template

If you wish to remove a template, activate it (the ![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/03000234.png)should appear behind the template name) and then click on �Delete�. 

Note: You cannot delete any of the templates that are installed by default with the Simulation module.

##### Importing Points 

Adding points from other templates

You do not need to define existing points again if you are creating a new template. Existing points can be imported quite easily from other templates with the �Import� function. Simply activate the template in which you wish to import the existing point(s) and click on the button �Import Points�. A new window will pop up. 

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/03000235_258x328.png)

From the drop down menu choose the template where you want to import points from. All points in that template are then shown in the window, select the point(s) you wish to import and click on the �import� button. If points from other templates also need to be imported select that template from the menu and repeat the steps above. To finish the import click on �ok�.

Of course, since default templates cannot be changed it is not possible either to import points into those templates.

##### Getting Started

The Anthropometric analysis tool allows for easy analysis and measurement based on pre-defined templates in both 3D and 2D views. To start the Anthropometric Analysis, go to the Simulation menu and choose Anthropometric Analysis.

Choosing the type of analysis

Before starting the analysis you first have to choose the analysis template you wish to use from the Anthropometric analysis pull down menu.

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/03000236_197x506.png)

If you do not find a template that coincides with your requirements you can create your own template as described in the creating a new template section. When you have chosen the template you wish to use, indicate the appropriate points, planes and/or measurements as described in the following sections.

###### p. Points of analysis

The points of the analysis pane provide you with a list of available points in the currently selected template. Points that have already been indicated on the images appear in black, the others appear in gray. The pane allows you to indicate, locate, edit or clear the points in the list. 

Indicating points

To place a point, first select it from the list and click on the �Indicate� button. You can indicate a point in both the 2D and the 3D views. Note that you first need to indicate the point before you can use any of the other option in the pane. When you have clicked the Indicate button, the description of the point will be displayed.

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/03000237_217x143.png)

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/03000238_369x66.png)

You can always move the points of the analysis after their indication. If you are not able to select the landmark points, first enable the right mouse mode by clicking on the Indicate button. Now you will be able to move the points.

Locating points

If you want to easily view the image on the location where a point was placed, highlight the point and click on �locate�. This will move both the axial and the sagittal view to the position where the point is located. Also a short description of the point you want to locate is given.

Clearing points

If you have misplaced a point you can easily remove it by selecting the point in the list and clicking on the �clear� button.

Editing points

After having indicated a point you can change its properties. If you click on the �edit� button you can easily change the color in which the point is shown on the images. You can also change the position of the point here by changing its coordinates.

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/03000239.png)

###### q. Planes of analysis

The pane for the planes of analysis allows you to indicate and edit the planes defined in the template.

Indicating planes

If you select the plane you want to view on the 3D image and click on �Indicate� a pop-up will ask you to indicate the different points that define the plane. If these points were already indicated the plane is shown automatically.

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/0300023A_217x143.png)

Editing planes

The �edit� option in the planes pane allows you to change the properties of the selected plane. You can choose its thickness, change its color, width and height and select what opacity the plane should have in the image.

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/0300023B_278x318.png)

###### r. Anthropometric 3D measurements

If your current template includes measurements, a list of these measurements is provided in the anthropometric measurements plane. These measurements can be easily indicated on the images.

Indicating measurements

Select the measurement you wish to indicate and click on the �indicate� button. A window will pop up to guide you through the placement of the points needed by the measurement. If these points where already indicated on the images, the measurement will have been made automatically. 

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/0300023C_217x143.png)

Viewing/exporting/printing measurement details

It is also possible to display a more detailed view of the measurements. Click on �Details� and a new window will appear showing all measurements with their pre-operative and post-operative values for both 2D and 3D. These values can be easily referenced to a set of normal; values that can be loaded with the �Load normal values� option. This overview can be printed out or exported to a csv-file for easy importing in other programs.

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/0300023D_448x406.png)

###### s. Sagittal plane

In case the CT images were not taken with the head of the patient at a 90-degree angle to the table, 2D and 3D measurements will be off. Therefore it is possible to adjust the position of the sagittal plane to account for this.

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/0300023E_217x52.png)

To change the position and direction of the sagittal plane, click on �Change� and draw the plane in your axial view. To return the plane to its original position, use the �Reset� option.

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/0200023F_306x265.jpg)
