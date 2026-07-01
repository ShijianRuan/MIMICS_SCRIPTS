# Medical vs. Research Edition

Terminology

The Medical edition of the Mimics Innovation Suite consists of the following medical device software components: Mimics Medical and 3-matic Medical. These products are medical devices and may therefore, within the limits of their intended use, be used by trained professionals to diagnose and plan patient treatments.

The Research edition of the Mimics Innovation Suite consists of the following software components: Mimics Research and 3-matic Research. These products are not medical devices and are not intended to be used for the diagnosis and/or treatment of patients. New modalities like X-Ray and 3D Ultrasound, and the latest cutting edge functionalities are included in the Research Edition only. These products are not commercially available in the Mimics Medical Edition in the US.

**Overview**

Depending on the license you purchased, you may have access to both editions on the same computer. As it is possible to open or modify files created in one edition in the other edition, we have implemented several notifications so that users are aware. If your workflow requires using the Medical edition only, we advise you not to install the Research edition.

The following features and behavior relate to the distinction between the Medical and the Research edition:

1\. Opening a project

2\. Importing and copy-pasting

3\. Preferences

4\. Recovery files

**Technical details**

**1\. Open project**

**General behavior: log messages**

To ensure traceability of Mimics projects across different versions and editions, the log panel will indicate the origin and the history of the project once it has been opened.

More specifically, the following information will be displayed:

Log message | Example | Explanation  
---|---|---  
Project created in <X> | Project created in Mimics Medical 17.0 | Name, edition and version number of the software in which this project was originally created.  
Project last modified in <X> | Project last modified in Mimics Research 17.0 | Name, edition and version number of the software in which this project was last modified.  
Edition: <X> | Edition: Research | The edition can be either Medical or Research. Medical indicates that this project and all objects within it were created in the Medical edition and were never modified outside of the Medical edition (with the exception of objects originating from generic imports, such as Import Part, Import STL, etc.). Research indicates that at least one object in the project was modified in the Research edition.   
  
**Remarks:**

Note that for a project with the �edition� field being Medical, the edition will be changed to Research when objects are directly imported or copy-pasted from Mimics or 3-matic Research.

As the Research and Medical editions were first introduced in version 17.0 for Mimics and version 9.0 for 3-matic, the �edition� field mentioned above will not be displayed for Mimics projects created or modified in or before Mimics 16.0 and 3-matic projects created or modified in or before 3-matic 8.0. The same applies for projects created in Mimics or 3-matic Medical in which an object from such a former version has been imported or copy-pasted.

For 3-matic projects created or modified in 3-matic 8.0 or earlier, the version in which the project was created will not be displayed.

**Special cases: pop-up messages**

The following scenarios will induce pop-up messages:

Scenario | Pop-up message  
---|---  
You open a project of which the Edition is Research, in Mimics Medical.  | �Mimics or 3-matic Research was used to create or modify an object in this project. How would you like to proceed?�   
You open a project of which the Edition is Medical, in Mimics Research. | �We recommend renaming your Mimics Medical project when modifying it in Mimics Research.� *  
  
* When you go to save the project in this scenario, the project name will be left blank. A �Save As� window will pop-up and allow you to define a new project name. This will help you to avoid overwriting a purely Medical project with a project that has now been modified in the Research edition.

**2\. Import and copy-paste**

**General behavior**

Importing a Mimics or 3-matic project or copy-pasting from Mimics or 3-matic, generates the following log message:

The following objects were imported from the Name.mcs project: <Entity1>, <Entity2>

Project created in <X>

Project last modified in <X>

Edition: <X>

The log message will contain the same information as when you open a project (Cf. section �Open Project�). Changes in edition classification related to data exchange follow the behavior described in the �Open Project� section as well.

If traceability is important to you and your process, we advise you not to disable logger messages. The logger messages ensure importing and copy-paste traceability.

**Special cases**

When importing or copy-pasting objects unknown to the application (e.g., X-Ray objects in Mimics Medical 21.0, Point Cloud objects in 3-matic Medical, etc.):

These objects will not be imported or copy-pasted

The log panel will show �Certain imported objects could not be loaded.�

When importing objects that originate from generic imports such as Import Part, Import STL, etc.:

No new warning or log message will be displayed

The edition of the project will not be updated

**3\. Preferences**

Preferences are not shared across editions. This specifically means that preferences for the Medical edition cannot be applied from a Research edition and vice versa.

**4\. Recovery files**

Recovery files are neither shared across editions nor with Mimics 16.0 or earlier versions.
