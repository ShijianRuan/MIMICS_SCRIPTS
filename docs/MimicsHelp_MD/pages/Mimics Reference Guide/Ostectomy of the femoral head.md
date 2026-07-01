### Ostectomy of the femoral head

To remove the femoral head we will use the polyplane cut from the 3D Tools menu.Go to 3D Tools> Cut > With Polyplane ![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/0200034C_25x22.jpg) in the menu . In the simulation dialog select the 3D model of the bone, Yellow. To perform the cut click once on the top of the femoral neck, turn the 3D and double click on the bottom. This will create a cutting plane as shown in the images below:

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/0300034D.png)  
---  
![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/0300034E_224x193.png) |  ![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/0300034F_224x192.png)  
  
The orientation of the cut can still be modified. Hover over the center of the red arrow, when the cursor changes into the reposition icon ![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/03000350_25x25.png), hold the left mouse button. By moving the mouse you can change the orientation of the cutting plane.

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/03000351_293x251.png)

Hold the left mouse button to change the orientation of the cutting plane

To finalize the cut the cutting plane should go completely through the bone. Therefore the depth needs to be increased. In the cut with PolyPlane dialog click on properties. In the properties dialog change the depth to 50 mm. 

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/03000352.png)  
---  
![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/03000353_221x170.png) |  ![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/03000354_200x172.png)  
  
Click on OK, to finish the cut.

The cut will create a new 3D model, PolyplanCut-Yellow. To split this model, go the 3D Tools> Split in the menu. In the Split dialog select the PolyplaneCut-yellow 3D model and select largest part. In this way you will only preserve the shaft of the femur. 

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/03000355_393x153.png)  
---  
![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/03000356_224x192.png) |  ![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/03000357_224x192.png)
