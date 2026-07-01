#### Import in ANSYS workbench

Open ANSYS workbench and drag the Finite Element Modeler component system to your project schematic. 

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/03000215_538x404.png)

Right click on Model, select Add Input Mesh. Select browse and chose Files of type �Mechanical APDL Input (*.cdb)� from the drop down menu. Now chose your ANSYS APDL Input mesh file (*.cdb) and click open.

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/03000216_538x404.png)

This will load your mesh in ANSYS workbench. Next, right click on Model and click on Edit to open the Finite Element Modeler window you�re your volume mesh. Check for the units system you are using in FE Modeler, this should be the same as that in Mimics for obtaining realistic results. 

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/03000217_552x298.png)

For simpler geometries it is sometimes better to use the automatic geometry creation tool. Here we will demonstrate how you can choose a region and assign boundary conditions on them. Select a few element�s faces using the �Select Element Faces� option  
and create components. 

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/03000218_554x297.png)

Right click on your selection and click on Add component option to save this surface for defining boundary conditions later. This surface will be used to apply load. We will add another surface component as fixity at the bottom of the femur as shown below. 

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/03000219_554x297.png)

Close the Finite Element Modeler window to save your data and move to next step in the analysis. 

Chose a standalone analysis system, example static structural, from the list of Analysis systems in workbench. Drag the model cell of the Finite element modeler to the model cell of your analysis system to transfer mesh data already processed in the last step. Update your project by clicking on Update Project button. 

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/0300021A_553x442.png)

Double click on the Setup option in your analysis system (in this case, the Static Structural cell). This will open the analysis setup window (Static Structural � Mechanical in this case) where we will define our boundary conditions and create an analysis. 

Your components created in Finite Element Modeler are available in Named Selections object. Select the component on which you wish to apply a boundary condition. You can right click on thee boundaries and select Insert to view the available boundary conditions. After selecting your choice, you can edit the parameters, like force magnitude etc. in the details view. In the figures below, a Pressure of 100Pa is applied on a region that we selected on the femur head, and a fixity boundary condition applied at the bottom face. 

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/0300021B_551x296.png)

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/0300021C_554x299.png)

If you have assigned materials in Mimics then an additional text file containing information about materials is exported with the mesh. To load this file, first insert a commands object in your outline as shown below. 

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/0300021D_554x299.png)

Next, right click on the Commands (APDL) object and select import. Select the text file exported with your mesh from Mimics. 

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/0300021E_554x297.png)

Now the materials that were defined in Mimics have been added to your analysis. 

Add solution visualization or post processing objects by right clicking on the solutions object and selecting the analysis of your choice. Launch the solver by clicking on solve button in the main menu. 

After the solver has converged, you can analyze your results by clicking on your post processing objects. An example solution is shown below. 

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/0300021F_561x303.png)
