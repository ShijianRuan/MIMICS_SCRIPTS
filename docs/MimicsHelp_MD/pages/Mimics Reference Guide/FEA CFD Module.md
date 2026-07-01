### FEA  
  
The FEA module provides a link to FEA (Finite Element Analysis) and CFD (Computational Fluid Dynamics) simulation.

  * Generate and export finite element meshes for FEA and CFD studies
  * Optimize by controlling global or local mesh sizes and quality
  * Create boundary surfaces
  * Perform CT gray value based heterogenous material assignment
  * Create multi-part assembly meshes


Supported formats:

  * Patran Neutral


  * Abaqus


  * ANSYS


  * Fluent


  * Nastran
  * COMSOL


### Starting the FEA module

All FEA functions are loaded immediately in Mimics after registration of the FEA module. When the module is registered, a few extra items are visible in the interface:

On the File -> Export menu:

  * An extra FEA menu, listing the different FEA functions. 


  * In the Export menu, new export file formats are added: Abaqus, ANSYS, Patran neutral, Fluent, Nastran and Comsol.


![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/030001D5_302x134.png)

In the Project management tab:

  * An extra FEA tab:


![](Mimics Reference Guide/../Resources/Images/Extra_FEA_tab.png)

Note: If you're not able to start the FEA module, it is possible that the correct password has not been used. Go to Help > Licenses and fill in the correct password.

The Mimics FEA Module enables you to link from scanned images to Finite Element Analysis (FEA) and Computational Fluid Dynamics (CFD) by exporting the files in the appropriate file format. You can calculate Parts based on the scanned images and prepare these surface meshes for Finite Element Analysis purposes. The Remesher in the FEA Module assures that you'll end up with the most optimal input for the pre-processor of your FEA software.

After converting the surface mesh to a volume mesh in the pre-processor, the volume mesh can be imported in Mimics again. Materials can then be assigned to the volume mesh, based on the Hounsfield Units in the scanned images or on the segmentation of the project.
