ECHO OFF
REM  This batch program registers Digital Video COM components
ECHO ON
C:\WINDOWS\system32\regsvr32 /S CMFileWriter.ax
C:\WINDOWS\system32\regsvr32 /S CMFileReader.ax
C:\WINDOWS\system32\regsvr32 /S CMReversePlay.ax
C:\WINDOWS\system32\regsvr32 /S CMVidCap.ax
C:\WINDOWS\system32\regsvr32 /S CMVidConfig.dll
C:\WINDOWS\system32\regsvr32 /S CMVideoInproc.dll
C:\WINDOWS\system32\regsvr32 /S CMVideoInprocps.dll
C:\WINDOWS\system32\regsvr32 /S CMVideoServerps.dll
C:\WINDOWS\system32\regsvr32 /S CMPanTiltZoom.ocx

CMVideoServer /RegServer
CMVideoServer /RegProfiles
CMVideoUser /I /S
