//------------------------------------------------------------------------------
// File: IVideoTrans.h
//
//------------------------------------------------------------------------------

#ifndef __IVIDEOTRANS__
#define __IVIDEOTRANS__

#ifdef __cplusplus
extern "C" {
#endif


//----------------------------------------------------------------------------
// INetSource's GUID
//

// {9C9BB46C-2163-42cd-90DF-9134CE46B5D8}
DEFINE_GUID(IID_IVideoTrans, 
0x9c9bb46c, 0x2163, 0x42cd, 0x90, 0xdf, 0x91, 0x34, 0xce, 0x46, 0xb5, 0xd8);



//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
// IAudioTrans
//----------------------------------------------------------------------------
DECLARE_INTERFACE_(IVideoTrans, IUnknown)
{

    STDMETHOD(put_SupportYUV) (THIS_
    				  BOOL bSupportYUV       /* [in] */	// Is support YUV
				 ) PURE;

    STDMETHOD(get_SupportYUV) (THIS_
    				  BOOL *bSupportYUV      /* [out] */	// Is support YUV
				 ) PURE;
	STDMETHOD(put_SupportDeinterlace) (THIS_
		BOOL bSupportDeinterlace             /* [in] */	   // Is support Deinterlace
		) PURE;
	STDMETHOD(get_SupportDeinterlace) (THIS_
		BOOL *bSupportDeinterlace            /* [out] */	   // Is support Deinterlace
		) PURE;
    STDMETHOD(put_AcceptOtherFilter) (THIS_
    				  BOOL bAcceptOtherFilter       /* [in] */	// accept other unknow filter ?
				 ) PURE;

    STDMETHOD(get_AcceptOtherFilter) (THIS_
    				  BOOL *bAcceptOtherFilter      /* [out] */	//
				 ) PURE;

    STDMETHOD(get_CurrentStreamTime) (THIS_
    				  DWORD *dwTime      /* [out] */	// get the current packet time
				 ) PURE;

    STDMETHOD(get_Location) (THIS_
    				  BSTR *lpszLocation      /* [out] */	
				 ) PURE;

    STDMETHOD(AccessToRegistry) (THIS_
    				  BOOL bRead      /* [out] */	
				 ) PURE;
};
//----------------------------------------------------------------------------

#ifdef __cplusplus
}
#endif

#endif // __IVIDEOTRANS__
