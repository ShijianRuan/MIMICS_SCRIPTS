#### Import a mesh in Abaqus

In Abaqus/CAE, go to File > Import > Model. Browse to the directory where you have saved your volume or surface mesh and select to import it.

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/03000204_554x295.png)

##### Convert a surface mesh to a volumetric mesh

In case you imported your surface mesh and you want to convert it to a volume mesh, choose one of the following workflows, according to the version of Abaqus you have:

###### o. Conversion in Abaqus 6.11:

Switch to the Mesh Module and choose to Edit the Mesh.

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/03000205_553x297.png)

This will open following dialog:

Here chose the Mesh option in Category and Convert tri to tet option in Method column. 

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/03000206_195x245.png)

Once you select these options, you can see the following message at the bottom of the 3D view window. Here, click Yes. 

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/03000207_214x74.png).

When the operation finishes, your surface mesh should be converted to a volume mesh.
