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
#define EC_TALK_NOTIFY	(EC_USER + 1002)
#define EC_ON_STOP_CONNECTION	(EC_USER + 1003)

//----------------------------------------------------------------------------
// INetSource's GUID
//

// {DBA2805A-4A2A-4550-9A93-7B2F99231029}
DEFINE_GUID(IID_INetSource, 
0xdba2805a, 0x4a2a, 0x4550, 0x9a, 0x93, 0x7b, 0x2f, 0x99, 0x23, 0x10, 0x29);

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
