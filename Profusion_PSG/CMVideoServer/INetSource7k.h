//------------------------------------------------------------------------------
// File: INetSource7k.h
//
//------------------------------------------------------------------------------


#ifndef __INETSOURCE7K__
#define __INETSOURCE7K__

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
// {307CB8CF-C1EA-4413-BCB1-3F6CB2D3F6DE}
DEFINE_GUID(IID_INetSource7K, 
0x307cb8cf, 0xc1ea, 0x4413, 0xbc, 0xb1, 0x3f, 0x6c, 0xb2, 0xd3, 0xf6, 0xde);

//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
// INetSource
//----------------------------------------------------------------------------
DECLARE_INTERFACE_(INetSource7K, IUnknown)
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
				 
	  STDMETHOD(put_MediaType) (THIS_
    				  DWORD dwMediaType       /* [in] */	
				 ) PURE;

    STDMETHOD(get_MediaType) (THIS_
    				  DWORD *dwMediaType      /* [out] */	
				 ) PURE;
				 
	  STDMETHOD(put_StreamIdx) (THIS_
    				  DWORD dwStreamIdx       /* [in] */	
				 ) PURE;

    STDMETHOD(get_StreamIdx) (THIS_
    				  DWORD *dwStreamIdx      /* [out] */	
				 ) PURE;
				 				 
    STDMETHOD(get_StreamNum) (THIS_
    				  DWORD *dwStreamNum      /* [out] */	
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

#endif // __INETSOURCE7K__
