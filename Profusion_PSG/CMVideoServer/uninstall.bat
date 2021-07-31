REM
REM CMVideoServer install script
REM
regsvr32 /s /U CMSplash.dll
regsvr32 /s /U CMFileWriter.ax
regsvr32 /s /U CMFileReader.ax
regsvr32 /s /U CMReversePlay.ax
regsvr32 /s /U CMVidCap.ax
regsvr32 /s /U CMVideoInproc.dll
regsvr32 /s /U CMVideoInprocps.dll
regsvr32 /s /U CMVideoServerps.dll
regsvr32 /s /U CMVidConfig.dll
regsvr32 /S /U CMPanTiltZoom.ocx
CMVideoServer /UnregServer