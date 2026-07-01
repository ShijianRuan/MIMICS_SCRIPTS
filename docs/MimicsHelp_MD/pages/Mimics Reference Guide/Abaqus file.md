#### The Abaqus file

There are some limitations for the format of the Abaqus file. The structure of the file should be:

*HEADING *NODE 1, 432.3656005859375, -245.597900390625, 54.94200134277344 2, 347.5975952148438, -158.1038970947266, 4.51170015335083 � *ELEMENT, elset=region0, type=C3D4 474604, 14874, 14869, 14873, 14872 474605, 14874, 14868, 14870, 14869 �  
---  
  
Three important rules to follow are:

  * The commands *HEADING, *NODE and *ELEMENT should be written in capitals.


  * The commands *HEADING and *NODE should not have any text or spaces behind them.


  * There should be an empty line between the *HEADING section and the *NODE command and between the *NODE section and the *ELEMENT command and after the *ELEMENT section.


##### Supported element types

The Mimics FEA module supports four types of Abaqus elements:

  * C3D4 and its variations (4-node linear tetrahedron)


  * C3D6 (6-node linear triangular prism)


  * C3D8 (8-node linear brick) 


  * C3D10 and its variations (10-node quadratic tetrahedron)


Remark: If you require other elements types, please let us know and we will try implement those elements in our future releases.

##### Supported Abaqus commands

The Mimics FEA module supports 5 Abaqus commands:

  * *HEADING


  * *NODE


  * *ELEMENT


  * *SOLID SECTION


  * *MATERIAL


The *SOLID SECTION and the *MATERIAL command are only used during export and are ignored during import.

The *ELEMENT command:

*ELEMENT, TYPE=<type>, ELSET=<name>

The *SOLID SECTION command:

*SOLID SECTION, ELSET=<name>, MATERIAL=<name>

The *MATERIAL command:

*MATERIAL, NAME=<name>

*DENSITY

<material density>

*ELASTIC

<E-modulus>, <Poisson>
