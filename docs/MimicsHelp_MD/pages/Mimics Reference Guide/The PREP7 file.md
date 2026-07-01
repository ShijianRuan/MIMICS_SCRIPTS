#### PREP7 file  
  
The ANSYS PREP7 file contains references to the nodes and elements files and material property definitions. An example of such a file (without material properties) can be found below:

/PREP7  ET,2,SOLID92  NRRANG,1,26407,1  
NREAD,'ANSYSTest_nodes_file',' ',' ' ERRANG,1,17317,1  
EREAD,'ANSYSTest_elements_file',' ',' '  
---  
  
This PREP7 file should be created manually. The easiest way is to copy the text to a new file and adapt it to your needs.

/PREP7  
This command indicates to Mimics that the file is a PREP7 file

ET,2,SOLID92

  
The first parameter (2) is the local element type and depends on your ANSYS project. The second parameter (SOLID92) is the type of the elements that is used in the mesh

NRRANG,1,26407,1  
The first parameter (1) is the index of the first node and will be 1 in most cases. The second parameter (26407) should be equal to the maximum node number. The third parameter (1) is the increment of the indices of the nodes and will be 1 in most cases.  
These parameters can be derived from the Node status. The node status dialog is evoked by executing the following commands:

  
NODES  
STAT  
The nodes status lists the maximum node number and the number of nodes defined. The index of the first node is equal to the maximum node number minus the number of nodes defined plus 1.

(index of first node =maximum node number - number of nodes defined + 1)

NREAD,'ANSYSTest_nodes_file',' ',' '

The first parameter (ANSYSTest_nodes_file) is the filename of the nodes file. The second parameter ( ) is the extension of the file (we suggest not to use an extension). The third parameter ( ) is the directory the file was written out to (we suggest to not use a directory)

ERRANG,1, 17317,1

The first parameter (1) is the index of the first node and will be 1 in most cases. The second parameter (17317) should be equal to the maximum element number. The third parameter (1) is the increment of the indices of the elements and will be 1 in most cases.

These parameters can be derived from the element status. The element status dialog is evoked by executing the following commands:

ELEM  
STAT  
The element status lists the maximum element number and the number of elements defined. The index of the first node is equal to the maximum element number minus the number of elements defined plus 1.

(index of first element =maximum element number - number of elements defined + 1)

EREAD,'ANSYSTest_elements_file',' ',' '

The first parameter (ANSYSTest_elements_file) is the filename of the elements file. The second parameter ( ) is the extension of the file (we suggest not to use an extension). The third parameter ( ) is the directory the file was written out to (we suggest to not use a directory)
