### Assign Material

You can assign piecewise linear-elastic materials to FEA meshes via the FEA/CFD menu or the FEA mesh tab. Mimics will calculate a gray value for each element. This gray value will then be used in further calculations to assign materials to the elements of the volumetric mesh.

Mimics uses an accurate method to assign gray values to elements by calculating exact intersections between voxels and subsequently determining the weighted average gray value for each element. While being accurate, care has been taken that the calculations can be performed efficiently.

Note: The units of Materials are defined according to the expressions you chose and will be thus different for different expressions.

**Material Assignment: Select Mesh**

When no volume mesh is selected this menu allows you to select a FEA mesh. When you have selected the correct mesh and click on the OK button, the material assignment window will be opened for the selected mesh.

The Sub-volumes, the different volumes of a non-manifold assembly, are displayed in the Select Mesh menu. You can select the main part to select all sub-volumes in case you want to perform the material assignment for the complete volume mesh. In case you want to assign material properties simply to a specific volume, then you can select them individually.

![](Mimics Reference Guide/../Resources/Images/MaterialAssignment_SelectMesh.png) |  ![](Mimics Reference Guide/../Resources/Images/FEA_PMTab.png)  
---|---  
  
**Note** : Ones the material assignment window is open you can easily switch between FEA Mesh objects by selecting the desired volume mesh in the PM tab.

**Material Assignment**

After selecting a volume mesh, the histogram and the corresponding elements will gradually appear, when the gray value for each element is calculated.

In the Elements Histogram, Mimics will show for each gray value the amount of elements that were assigned that particular value. After the gray value calculation, each element has its own gray value based on the image data set. By settings the Material expressions this gray value can be convert into the desired material properties. The color of the elements will be determined according to following color scale:

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/020001F1_449x67.jpg)

You can choose between four different material assignment methods. Depending on the method, the view of the tool changes.

![](Mimics Reference Guide/../Resources/Images/FEACFD_AssignMaterial.png)

**Note** : Most FEA software do not allow you to enter a density with a negative value, so make sure you choose your expression accordingly or adjust the values manually in the material editor.

**Warning** : The following default values are used when leaving one of the checkboxes on OFF. Density = 0, Young�s modulus = 0 and Poisson Coefficient = 0.5.
