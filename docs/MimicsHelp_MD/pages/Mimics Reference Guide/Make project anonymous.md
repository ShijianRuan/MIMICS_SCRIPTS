### Anonymize   
  
The �Anonymize� tool allows you to anonymize the images in your project. The tool follows NEMA�s PS3.15 standard, Appendix E [(http://dicom.nema.org/medical/dicom/current/output/chtml/part15/chapter_E.html)](<http://dicom.nema.org/medical/dicom/current/output/chtml/part15/chapter_E.html>).

By default, the tool anonymizes with the Basic Application Level Confidentiality Profile. However, checkboxes in the dialog of the tool allow you to retain more data or clean more data as desired.

![](Mimics Reference Guide/../Resources/Images/anonymize_tool_1.png)

The right pane of the �Anonymize� dialog shows three tabs with DICOM tags: �Modify�, �Keep� and �All�.

![](Mimics Reference Guide/../Resources/Images/anonymize_tool_2.png)

The �**Modify** � tab shows all the tags that will be modified by the anonymization, the �**Keep** � tab shows all the tags that will be retained, and the �**All** � tab shows all tags in the data.

The �**Modify** � tab shows per tag the current value of the tag (= before anonymization) and the value that it will be replaced with after anonymization. If you want to use the proposed anonymization, press �Anonymize� to apply. Alternatively, if desired you can change any of the proposed values in the �Anonymized Value� column to a custom value. To do so, double click on some value (or empty cell) in the �Anonymized Value� column, fill in any custom value, and press Enter.

![](Mimics Reference Guide/../Resources/Images/anonymize_tool_3.png)

In case your project consists of multiple image sets (as seen in the Images Project Management tab), anonymization will by default only be applied on the active image set. To apply it on all the image sets, use the checkbox �Anonymize All Images�.

Note that if you press �Close�, the anonymization will not be applied.

_Note_ : The Anonymize tool does not support the anonymization of DICOM Ultrasound images.
