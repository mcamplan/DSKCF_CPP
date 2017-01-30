::Example on how to use the compiled C++ dskcf (bat script for windows platform)
::BE SURE YOU DEFINE CORRECTLY THE NEXT VARIABLES FOR THIS BAT SCRIPT

::Set top directory and a string....
set topResultsDir=C:\myDevelopments\JRTIP_DSKCF_releaseCPP\resDir
set topResultsDirSTRING=C:/myDevelopments/JRTIP_DSKCF_releaseCPP/resDir
::Set data directory
set dataDirectory=C:/myDevelopments/JRTIP_DSKCF_releaseCPP/dataPrincetonValidation
::Set compiled file
set myDSKCFexe=C:\myDevelopments\ds-kcf-cJournalVersion\compiledDSKCF_VS12\Release\DSKCFcpp.exe
::Set 



::generate directories for all the sequences in the validation set
::echo %topResultsDir%
::pause
mkdir %topResultsDir%\bear_front
mkdir %topResultsDir%\child_no1
mkdir %topResultsDir%\zcup_move_1
mkdir %topResultsDir%\face_occ5
mkdir %topResultsDir%\new_ex_occ4

:: NOW USE DSKCF, HERE SAVING ALSO IMAGING ON THE DISK AND DISPLAYING AS WELL.....THIS MAY SLOW DOWN A BIT THE TRACKER.....

%myDSKCFexe% -b  178,162,121,156  -d  -e %topResultsDirSTRING%/bear_front/ -o %topResultsDirSTRING%/bear_front/bear_front.txt -s %dataDirectory%/bear_front/rgb -i /%%.06d.png --depth_sequence %dataDirectory%/bear_front/depth --depth_image_name_expansion /%%.06d.png
%myDSKCFexe% -b 238,186,58,156 -d   -e %topResultsDirSTRING%/child_no1/ -o %topResultsDirSTRING%/child_no1/child_no1.txt -s %dataDirectory%/child_no1/rgb -i /%%.06d.png --depth_sequence %dataDirectory%/child_no1/depth --depth_image_name_expansion /%%.06d.png
%myDSKCFexe% -b 236,295,73,139  -d  -e %topResultsDirSTRING%/zcup_move_1/ -o %topResultsDirSTRING%/zcup_move_1/zcup_move_1.txt -s %dataDirectory%/zcup_move_1/rgb -i /%%.06d.png --depth_sequence %dataDirectory%/zcup_move_1/depth --depth_image_name_expansion /%%.06d.png
%myDSKCFexe% -b 242,186,79,101  -d  -e %topResultsDirSTRING%/face_occ5/ -o %topResultsDirSTRING%/face_occ5/face_occ5.txt -s %dataDirectory%/face_occ5/rgb -i /%%.06d.png --depth_sequence %dataDirectory%/face_occ5/depth --depth_image_name_expansion /%%.06d.png
%myDSKCFexe% -b 109,214,84,266  -d  -e %topResultsDirSTRING%/new_ex_occ4/ -o %topResultsDirSTRING%/new_ex_occ4/new_ex_occ4.txt -s %dataDirectory%/new_ex_occ4/rgb -i /%%.06d.png --depth_sequence %dataDirectory%/new_ex_occ4/depth --depth_image_name_expansion /%%.06d.png


::pause

