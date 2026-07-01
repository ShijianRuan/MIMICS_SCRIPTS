#### Import a mesh in ANSYS Mechanical APDL

Your volume or surface mesh can be imported in ANSYS by going to the File menu in ANSYS and choosing Read Input From. 

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/0300020D_454x350.png)

##### Convert an Element Based surface mesh to a volumetric mesh

In ANSYS select File | Read input from and load the file

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/0300020E_448x349.png)

The surface element type is Shell93 by default.

Note: If desired you can change the surface element type with the command ET, ITYPE, Ename, KOP1, KOP2, KOP3, KOP4, KOP5, KOP6, INOPR (e.g. et,1,Mesh200)

Add a solid element type to generate your volumetric mesh, you can again do this with the ET command e.g. ET,2,SOLID92. Or in the main menu select Preprocessor | Element type | Add/Edit/delete and add a solid element type.

If you want to load your volume mesh back into Mimics for material assignment you should use one of the following element types:

  * SOLID72, SOLID185 (linear tetrahedron)


  * SOLID92, SOLID187 (quadratic tetrahedron)


  * SOLID185 (linear hexahedron)


If you require other elements types, please let us know and we will try implement those elements in our future releases.

To generate the volume mesh you should use the command FVMesh or you can execute this command from the main menu Preprocessor | Meshing | Mesh | Tet Mesh From | Area Elements. 

##### Convert an Area Based surface mesh to a volumetric mesh

First select an element type to which you want to assign the surface mesh. You can do this with the ET command e.g. ET,1,SHELL93. Or in the main menu select Preprocessor | Element type | Add/Edit/delete and add a surface element type.

To mesh the areas to elements, go to Main Menu > Preprocessor > Meshing > MeshTool. In the MeshTool check smart size and put it to coarse. Select as mesh Areas and as Shape Tri and free. 

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/0300020F_164x600.png)

Click on Mesh, the Mesh areas dialog pops up. In this dialog click on pick all.

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/03000210.png)

To release all associations between the current solid model and finite element model, execute the command, MODMSH,detach.

Add a solid element type to generate your volumetric mesh, you can do this with the ET command e.g. ET,2,SOLID92. Or in the main menu select Preprocessor | Element type | Add/Edit/delete and add a solid element type.

If you want to load your volume mesh back into Mimics for material assignment you should use one of the following element types:

  * SOLID72 or SOLID185 (linear tetrahedron)


  * SOLID92 or SOLID187(quadratic tetrahedron)


If you require other elements types, please let us know and we will try to implement those elements in our future releases.

To generate the volume mesh you should use the command FVMesh or you can execute this command from the main menu Preprocessor | Meshing | Mesh | Tet Mesh From | Area Elements. 
