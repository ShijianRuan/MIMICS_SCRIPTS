#### Change the Abaqus pattern in Simmetrix

To make sure that you write out an Abaqus file in Simmetrix that can be imported in Mimics you will have to change the default Abaqus pattern file in the Simmetrix software. This pattern file will determine how an Abaqus file is written out exactly. To do this, first go to the export/case/abaqus folder in the root directory of the SimAppS folder and make a backup of the abaqus.sxp file in this folder.

After this open the abaqus.sxp file in a text editor (e.g. notepad) and change the contents of the file to:

sxp 0 # abaqus.sxp created 28-Jan-2004 jat # Write abaqus input file from input mesh # Use mesh node ids required = "mesh" precision = 16 showpoint = 1 mfaceNormals = "off" # (runs faster when "off") mregionIdOffset = 1 # Create mesh nodes mesh { nodeMessage = function:makeNodes() log:header = "$nodeMessage" } # Write first header header = "*HEADING\n" # Write mesh nodes mesh/mnodes { header = "*NODE\n" item = "$id, $x, $y, $z" } # Set element-code based on mesh degree map = "<1 'C3D4'> <2 'C3D10'>" element-code = function:mapI2S($meshDegree,$map,"C3D???") # Write element sets, 1 for each model region # For now, only support linear tetrahedral gmodel/gregions/* { header = "*ELEMENT, elset=region$tag, type=$element-code\n" mregions { item = "$id, $(mnodes[*]/id)" } # end context (mregions/*) } # end context (gmodel/gregions/*) # Remove mesh nodes mesh { nodeMessage = function:removeNodes() log:header = "$nodeMessage" }  
---
