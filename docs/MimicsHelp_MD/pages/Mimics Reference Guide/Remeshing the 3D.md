### Remeshing the Part

In this step the Part needs to be remeshed to be optimal for FEA purposes. You will notice that there are two Parts, select the Yellow 2 part. The FemurShaft model will be used in the non-manifold assembly tutorial. To export the Part to 3-matic, go to the FEA -> Remesh in the menu. This will bring up the following dialog:

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/030002F2.png)

Select the Yellow 2 Part and click on OK.

To get the optimal result it is common to follow the next steps in 3-matic:

\- Define shape parameters

\- Inspect the quality of the surface mesh

\- Reduce the details of the anatomical part

\- Remesh the surface elements

\- Generate the volume mesh

\- Visualize volume elements

\- Analyze Mesh quality

For details on how to get the optimal mesh for your Part see the **Chapter 5: Remesh** of the Tutorials in 3-matic. You can find it under **Help** -> **Tutorial..** in 3-matic.

When you are satisfied with the quality of the mesh copy your object by selecting your object and pressing Ctrl+C on your keyboard. Open your Mimics window and paste your object there by pressing Ctrl+V. The volume mesh will be available in the FEA mesh tab in the project management section. These meshes can then be exported to your FEA software.
