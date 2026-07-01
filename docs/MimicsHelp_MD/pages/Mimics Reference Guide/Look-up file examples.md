# Lookup file examples

Example lookup file for Gray value based material assignment:

![](Mimics Reference Guide/../Resources/Images/LookupFileExample_GV1.png)  
---  
![](Mimics Reference Guide/../Resources/Images/LookupFileExample_GV2.png)  
  
<?xml version="1.0" encoding="UTF-8"?>

<LookupTable>

<Header>

<Version>

<Major>2</Major>

<Minor>0</Minor>

</Version>

<Units>Hounsfield</Units>

</Header>

<Table>

<Interval>

> <Start>-1024</Start>
> 
> <NumberOfMaterials>1</NumberOfMaterials>
> 
> <Density>
> 
> <A>0.35</A>
> 
> <B>0</B>
> 
> </Density>
> 
> <EModulus>
> 
> <A>4000</A>
> 
> <B>0</B>
> 
> <C>1</C>
> 
> <D>0</D>
> 
> <E>1</E>
> 
> </EModulus>
> 
> <PoissonCoefficient>
> 
> <A>0.3</A>
> 
> <B>0</B>
> 
> <C>1</C>
> 
> <D>0</D>
> 
> <E>1</E>
> 
> </PoissonCoefficient>

</Interval>

<Interval>

> <Start>700</Start>
> 
> <NumberOfMaterials>10</NumberOfMaterials>
> 
> <Density>
> 
> <PixelValue1> 700</PixelValue1>
> 
> <Coefficient1> 1.2</Coefficient1>
> 
> <PixelValue2> 1500</PixelValue2>
> 
> <Coefficient2> 1.92</Coefficient2>
> 
> </Density>
> 
> <EModulus>
> 
> <A>-22000</A>
> 
> <B>24000</B>
> 
> <C>1</C>
> 
> <D>0</D>
> 
> <E>1</E>
> 
> </EModulus>
> 
> <PoissonCoefficient>
> 
> <A>0.3</A>
> 
> <B>0</B>
> 
> <C>1</C>
> 
> <D>0</D>
> 
> <E>1</E>
> 
> </PoissonCoefficient>

</Interval>

</Table>

<TetrahedronsOutsideMask>

<Density>1.92</Density><Emod>24</Emod><PR>0.3e0</PR>

</TetrahedronsOutsideMask>

</LookupTable>

Example lookup file for Mask based material assignment:

![](Mimics Reference Guide/../Resources/Images/LookupFileExample_MaskBased1.png)  
---  
![](Mimics Reference Guide/../Resources/Images/LookupFileExample_MaskBased2.png)  
  
<?xml version="1.0" encoding="UTF-8"?>

<LookupTable>

<Header>

<Version>

<Major>2</Major>

<Minor>0</Minor>

</Version>

<Units>Mask</Units>

</Header>

<Table>

<Interval>

> <Mask>Trabecular</Mask>
> 
> <NumberOfMaterials>1</NumberOfMaterials>
> 
> <Density>
> 
> <A>0.35</A>
> 
> <B>0</B>
> 
> </Density>
> 
> <EModulus>
> 
> <A>4000</A>
> 
> <B>0</B>
> 
> <C>1</C>
> 
> <D>0</D>
> 
> <E>1</E>
> 
> </EModulus>
> 
> <PoissonCoefficient>
> 
> <A>0.3</A>
> 
> <B>0</B>
> 
> <C>1</C>
> 
> <D>0</D>
> 
> <E>1</E>
> 
> </PoissonCoefficient>

</Interval>

<Interval>

> <Mask>Cortical</Mask>
> 
> <NumberOfMaterials>10</NumberOfMaterials>
> 
> <Density>
> 
> <PixelValue1> 700</PixelValue1>
> 
> <Coefficient1> 1.2</Coefficient1>
> 
> <PixelValue2> 1500</PixelValue2>
> 
> <Coefficient2> 1.92</Coefficient2>
> 
> </Density>
> 
> <EModulus>
> 
> <A>-22000</A>
> 
> <B>24000</B>
> 
> <C>1</C>
> 
> <D>0</D>
> 
> <E>1</E>
> 
> </EModulus>
> 
> <PoissonCoefficient>
> 
> <A>0.3</A>
> 
> <B>0</B>
> 
> <C>1</C>
> 
> <D>0</D>
> 
> <E>1</E>
> 
> </PoissonCoefficient>

</Interval>

</Table>

<TetrahedronsOutsideMask>

<Density>1.92</Density><Emod>24</Emod><PR>0.3e0</PR>

</TetrahedronsOutsideMask>

</LookupTable>

Examples Lookup file Mimics 17.0 and older versions

**Note** : Mimics 17.0 and older versions are still supported by the newest Mimics releases. The values shown in the older format of the lookup files are directly used in the Material Editor menu and are not showed in the material expressions.

Example lookup file for Hounsfield Unit based material assignment (Mimics 17.0 and older versions):

<?xml version="1.0" encoding="UTF-8"?>

<LookupTable>

<Header>

<Version>

<Major>1</Major>

<Minor>0</Minor>

</Version>

<Units>Hounsfield</Units>

</Header>

<Table>

<Interval><Start> -100</Start><Density> 1.0 </Density><Emod> 1904000000</Emod><PR>0.3e0</PR></Interval>

<Interval><Start> 0 </Start><Density> 1.125 </Density><Emod> 2309700000</Emod><PR>0.3e0</PR></Interval>

<Interval><Start> 500 </Start><Density> 1.25 </Density><Emod> 2745400000</Emod><PR>0.3e0</PR></Interval>

</Table>

<TetrahedronsOutsideMask>

<Density>2000</Density><Emod>17583000000</Emod><PR>0.3e0</PR>

</TetrahedronsOutsideMask>

</LookupTable>

This Lookup file specifies 3 materials:

  * Material 1 contains all elements with HU (Hounsfield Unit) between -100 and 0. That material is assigned a density of 1 , Young�s modulus of 1.904e9 and Poisson�s ratio of 0.3.


  * Material 2 contains all elements with HU between 0 and 500. That material is assigned a density of 1.125, Young�s modulus of 2.3097e9 and Poisson�s ratio of 0.3.


  * Material 3 contains all elements with HU above 500. That material is assigned a density of 1.25, Young�s modulus of 2.7454e9 and Poisson�s ratio of 0.3.


Example lookup file for Mask based material assignment (Mimics 17.0 and older versions):

<?xml version="1.0" encoding="UTF-8"?>

<LookupTable>

<Header>

<Version>

<Major>1</Major>

<Minor>1</Minor>

</Version>

<Units>Mask</Units>

</Header>

<Table>

<Interval><Mask>Green</Mask><Density> 0.0 </Density><Emod>1.0e9</Emod><PR>0.3e0</PR></Interval>

<Interval><Mask>Yellow</Mask><Density> 1.0 </Density><Emod>1.1e9</Emod><PR>0.3e0</PR></Interval>

<Interval><Mask>Cyan</Mask><Density> 1.1 </Density><Emod>1.2e9</Emod><PR>0.3e0</PR></Interval>

</Table>

<ExclusionMaterial>

<value><Density> 1.3 </Density><Emod>1.0e9</Emod><PR>0.3e0</PR></value>

</ExclusionMaterial>

</LookupTable>

The following table introduces the tags used in Lookup file to define material properties. 

Interval |  Defines the starting of Hounsfield unit or Gray value range. It can also contain a Mask name from the project.  
---|---  
Density |  Defines the density for the above defined interval.  
Emod |  Contains value of E modulus  
PR |  Contains value of poissons ratio  
  
Note: You can choose between the type of lookup file by specifying "Hounsfield", "Gray value", or �Mask� as the Units section of your Lookup file.

Several example lookup files are available in the MedData folder of the local installation directory.
