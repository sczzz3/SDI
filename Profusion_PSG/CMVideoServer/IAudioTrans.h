//------------------------------------------------------------------------------
// File: IAudioTrans.h
//
//------------------------------------------------------------------------------


#ifndef __IAUDIOTRANS__
#define __IAUDIOTRANS__

#ifdef __cplusplus
extern "C" {
#endif


//----------------------------------------------------------------------------
// INetSource's GUID
//

// {C9AAF633-9C0C-4094-84F0-2F9BF7784F1E}
DEFINE_GUID(IID_IAudioTrans, 
0xc9aaf633, 0x9c0c, 0x4094, 0x84, 0xf0, 0x2f, 0x9b, 0xf7, 0x78, 0x4f, 0x1e);


//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
// IAudioTrans
//----------------------------------------------------------------------------
DECLARE_INTERFACE_(IAudioTrans, IUnknown)
{
    STDMETHOD(put_AcceptOtherFilter) (THIS_
    				  BOOL bAcceptOtherFilter       /* [in] */	// accept other unknow filter ?
				 ) PURE;

    STDMETHOD(get_AcceptOtherFilter) (THIS_
    				  BOOL *bAcceptOtherFilter      /* [out] */	//
				 ) PURE;

    STDMETHOD(get_CurrentStreamTime) (THIS_
    				  DWORD *dwTime      /* [out] */	// get the current packet time 
				 ) PURE;

    STDMETHOD(AccessToRegistry) (THIS_
    				  BOOL bRead      /* [out] */	
				 ) PURE;
};
//----------------------------------------------------------------------------

#ifdef __cplusplus
}
#endif

#endif // __IAUDIOTRANS__
