# Lookup files

A lookup file can be used to assign material properties according to user specific ranges and material values. A lookup file is in XML format that defines which intervals are used to divide the range of gray values (look-up files can also be expressed in Hounsfield units or with Mask names in your project). Each interval can be assigned a specific density, Young�s modulus and Poisson�s ratio values.Material expressions can be saved in a lookup file with the **Save Lookup file** button. To automatically fill in material expressions with a lookup file use the **Load Lookup file** button.

It is possible to limit the assignment of material properties to only the tetrahedrons that lie inside a specified mask by using the **Limit to Mask** feature.

![](Mimics Reference Guide/../Resources/Images/FEACFD_AssignMaterial_LookupFile.png)

Sometimes, the ranges defined in the lookup table do not cover all the tetrahedrons of an FEA mesh. All such tetrahedrons are assigned the color Gray. As explained below, you may assign material properties to these tetrahedrons using the �TetrahedronsOutsideMask� tag or you can also define material properties to such materials directly from the Material Editor tab. 

The format of the lookup table is very simple. The first line specifies the xml format of the file. In the header of the xml file, the version of the file can be specified (1.0 for the moment). Next, the Units in which the ranges are defined is specified, this can be Hounsfield, Gray value or Mask. Then, under the Table header, the gray value/ Housfield Unit intervals or Mask name is defined along with the corresponding Density, Youngs modulus and Poisson�s ratio values. Next, material properties of tetrahedrons that do not lie in the defined Hounsfield unit/gray value ranges or in the defined Masks are provided using in a different section under the tags �TetrahedronsOutsideMask�. The following example shows the structure of the lookup file.

![](Mimics Reference Guide/../Resources/Images/FEACFD_AssignMaterial_LookupFile2_638x500.png)
