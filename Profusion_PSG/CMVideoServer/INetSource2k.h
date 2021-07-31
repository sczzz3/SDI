//------------------------------------------------------------------------------
// File: INetSource.h
//
//------------------------------------------------------------------------------


#ifndef __INETSOURCE__
#define __INETSOURCE__

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

// {3E02D508-8381-498b-8D7C-269B324BA004}
DEFINE_GUID(IID_INetSource, 
0x3e02d508, 0x8381, 0x498b, 0x8d, 0x7c, 0x26, 0x9b, 0x32, 0x4b, 0xa0, 0x4);


//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
// INetSource
//----------------------------------------------------------------------------
DECLARE_INTERFACE_(INetSource, IUnknown)
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

    STDMETHOD(put_UserName) (THIS_
    				  BSTR szUserName      /* [out] */	
				 ) PURE;

    STDMETHOD(put_Password) (THIS_
    				  BSTR szPassword      /* [out] */	
				 ) PURE;

    STDMETHOD(put_DisplaySystrayIcon) (THIS_
    				  BOOL bDisplaySystrayIcon      /* [out] */	
				 ) PURE;

	STDMETHOD(put_Camera) (THIS_
    				  DWORD dwCamera       /* [in] */	
				 ) PURE;

    STDMETHOD(get_Camera) (THIS_
    				  DWORD *pdwCamera      /* [out] */	
				 ) PURE;
	STDMETHOD(put_VideoSize) (THIS_
    				  DWORD dwVideoSize       /* [in] */	
				 ) PURE;

    STDMETHOD(get_VideoSize) (THIS_
    				  DWORD *pdwVideoSize      /* [out] */
				 ) PURE;
	STDMETHOD(put_Quality) (THIS_
    				  DWORD dwQuality       /* [in] */	
				 ) PURE;

    STDMETHOD(get_Quality) (THIS_
    				  DWORD *pdwQuality      /* [out] */
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

#endif // __INETSOURCE__
