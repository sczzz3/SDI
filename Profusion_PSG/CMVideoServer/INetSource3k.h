//------------------------------------------------------------------------------
// File: INetSource3K.h
//
//------------------------------------------------------------------------------


#ifndef __INETSOURCE3K__
#define __INETSOURCE3K__

#ifdef __cplusplus
extern "C" {
#endif

// param1 is Sample Per second, param2 low word is channel numbuer and high word is Bits per sample
#define EC_AUDIO_CHANGE (EC_USER + 1000)  
// param1 is the new codec
#define EC_VIDEO_CODEC_CHANGE (EC_USER + 1001)
#define EC_ON_STOP_CONNECTION	(EC_USER + 1003)

//----------------------------------------------------------------------------
// INetSource's GUID
//

// {CF299EEA-7597-433c-A5EF-2864292D0099}
DEFINE_GUID(IID_INetSource3K, 
0xcf299eea, 0x7597, 0x433c, 0xa5, 0xef, 0x28, 0x64, 0x29, 0x2d, 0x0, 0x99);

//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
// INetSource
//----------------------------------------------------------------------------
DECLARE_INTERFACE_(INetSource3K, IUnknown)
{

    STDMETHOD(put_DropAudioSample) (THIS_
    				  BOOL bDropAudio       /* [in] */	
				 ) PURE;

    STDMETHOD(get_DropAudioSample) (THIS_
    				  BOOL *bDropAudio      /* [out] */	
				 ) PURE;

    STDMETHOD(put_ReadFromRegistry) (THIS_
    				  BOOL bReadFromRegistry       /* [in] */	
				 ) PURE;

    STDMETHOD(get_ReadFromRegistry) (THIS_
    				  BOOL *bReadFromRegistry      /* [out] */	
				 ) PURE;

	STDMETHOD(put_Protocol) (THIS_
    				  DWORD dwProtocol       /* [in] */	
				 ) PURE;

    STDMETHOD(get_Protocol) (THIS_
    				  DWORD *dwProtocol      /* [out] */	
				 ) PURE;
    STDMETHOD(put_UserName) (THIS_
    				  BSTR szUserName      /* [out] */	
				 ) PURE;

    STDMETHOD(put_Password) (THIS_
    				  BSTR szPassword      /* [out] */	
				 ) PURE;

    STDMETHOD(put_DisplaySystrayIcon) (THIS_
    				  BOOL bDisplaySystrayIcon      /* [out] */	
				 ) PURE;
				 
    STDMETHOD(AccessToRegistry) (THIS_
    				  BOOL bRead      /* [out] */	
				 ) PURE;
    STDMETHOD(put_PopupErrorMsg) (THIS_
    				  BOOL bPopupMsg  /* [in] */	
				 ) PURE;
    STDMETHOD(get_PopupErrorMsg) (THIS_
    				  BOOL * bPopupMsg  /* [out] */	
				 ) PURE;				 
};
//----------------------------------------------------------------------------

#ifdef __cplusplus
}
#endif

#endif // __INETSOURCE3K__
