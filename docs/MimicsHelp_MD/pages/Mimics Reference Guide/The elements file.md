#### Elements file

The elements file contains the definitions for all elements. It�s strongly advised to write out the elements file in the LONG format. The file should be written out with following command:

EWRITE, Fname, Ext, Dir, KAPPND, Format 

With parameters:

  * Fname: File name (32 characters maximum)


  * Ext: File name extension (8 characters maximum)


  * Dir: Directory name (64 characters maximum)


  * KAPPND: Append key:


  * 0 - Rewind file before the write operation


  * 1 - Append data to the end of the existing file


  * Format: Format key:


  * SHORT - I6 format (the default)


  * LONG1 - I8 format


Note: We suggest to write out the elements file without an extension.

Note: We suggest to write out the elements file in the LONG format. If you have more than 99.999 nodes, you have to write out in the LONG format or Mimics will refuse to import the files.
