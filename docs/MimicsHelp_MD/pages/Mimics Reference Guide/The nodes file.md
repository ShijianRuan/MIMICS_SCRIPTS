#### Nodes file

The nodes file contains the coordinates for all nodes. The file can be written out in ANSYS by going to: Main Menu > Preprocessor > Modeling > Create > Nodes > Write Node File.

You can also write out the nodes file with following command:

NWRITE, Fname, Ext, Dir, KAPPND 

With parameters:

  * Fname: File name (32 characters maximum)


  * Ext: File name extension (8 characters maximum)


  * Dir: Directory name (64 characters maximum)


  * KAPPND: Append key:


  * 0 - Rewind file before the write operation


  * 1 - Append data to the end of the existing file


Note: We suggest to write out the nodes file without an extension.
