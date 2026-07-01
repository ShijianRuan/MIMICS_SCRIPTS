### Material Assignment

When you have created a volumetric mesh from your remeshed object, you can perform the material assignment in Mimics. You can see the mesh listed in the FEA mesh tab. 

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/03000310_236x224.png)

Note: We will use gray values for this tutorial, so if you are working in Hounsfield units, please change this by going to the Edit menu, choose Preferences and change the Pixel Unit in the General tab.

With the FEA mesh of the Femur selected, click on the Materials button. Mimics will display a message that the gray values for this mesh have to be calculated before you can do a material assignment. Choose �Yes� to continue. After the calculation you will see following dialog box:

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/03000311_739x579.png)

Mimics shows for each gray value the amount of elements that were assigned that particular value. We will then convert this gray value to material properties. In this tutorial we will use the uniform method. 

STEP A: If the Gray value based method is not selected, click on the radio button next to Uniform. 

STEP B: Enter the number of materials in the edit box. We will use 10 materials for this tutorial. The FEA module will now divide the range of gray values that occur in the volume mesh into 10 equally sized intervals that each represents a material. You can see this discretization by choosing the Materials histogram. Select Limit to Mask: Green 2. The limit assignment to mask intercepts the deviation in the boundary elements due to the partial volume effect. As boundary voxels typically represent multiple tissues by excluding these voxels, the material assignment will become more accurate.

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/03000312.png)

STEP C: Enter a density expression to convert the gray value of each material to a density. For this tutorial we will use following expression: Density = -13.4 + 1017 * Gray value.

STEP D: Choose to write out only the Young's modulusmaterial properties in the exported file by deselecting the selection boxes before Density and Poisson Coefficient. We will use following expression for the Young's modulus: E-Modulus = -388.8 + 5925 * Density.

STEP E: Check the values for the materials that will be assigned in the material editor:

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/03000313_697x559.png)

STEP F: Press the Apply button to assign the materials to the FEA mesh. The elements of the FEA mesh will be colored according to their materials:

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/03000314_449x404.png)

This volumetric mesh can then be exported together with the material assignment (in this case only the E-Modulus).

Note: It is also possible to use different expressions for different ranges or different masks material assignment. For more detail on this check the section on Material Assignment using Lookup Files. 
