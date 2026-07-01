#### Export a volumetric file to ANSYS Mechanical APDL

After you have calculated a volume mesh suitable for FEA purposes, you can export this mesh to an ANSYS file. To do this, go to the Export menu and choose ANSYS. Select your volume mesh to be exported, set your Output Format as ANSYS Preprocessor files and click on the OK button.

##### Export a surface file to ANSYS Mechanical APDL

If you want to create a volume mesh in ANSYS, starting from a surface mesh created in Mimics, you can export this surface mesh as an ANSYS file. To do this, go to the Export menu and choose ANSYS. This will open following interface:

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/0300020C_440x412.png)

Mimics can export your remeshed Part to ANSYS as an Area-based or an Element-based file:

  * Element-based: The part is exported as a mesh, having triangles as elements. To generate the volume mesh in ANSYS, the FVMESH command should be used. 


  * Area-based: Each triangle is exported as a separate face into the ANSYS-file. Importing this file in ANSYS will result in a remeshing of the file, and will lose the original triangulation.


We advise you to use the Element-based export as this will preserve the obtained quality of the mesh. To export add the object to the list, choose the appropriate ANSYS File format and click on the OK button.
