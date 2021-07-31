//------------------------------------------------------------------------------
// File: ITextOnVideo.h
//
//------------------------------------------------------------------------------

#ifndef __ITEXTONVIDEO__
#define __ITEXTONVIDEO__

#ifdef __cplusplus
extern "C" {
#endif


//----------------------------------------------------------------------------
// INetSource's GUID
//

// {ABF0CC57-1B9D-47ff-8FC7-4812CF5198B6}
DEFINE_GUID(IID_ITextOnVideo, 
0xabf0cc57, 0x1b9d, 0x47ff, 0x8f, 0xc7, 0x48, 0x12, 0xcf, 0x51, 0x98, 0xb6);

// {9E7524F0-BA91-4445-9BF4-7F53C954A827}
DEFINE_GUID(IID_IFontProperty, 
0x9e7524f0, 0xba91, 0x4445, 0x9b, 0xf4, 0x7f, 0x53, 0xc9, 0x54, 0xa8, 0x27);


//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
// ITextOnVideo
//----------------------------------------------------------------------------
DECLARE_INTERFACE_(ITextOnVideo, IUnknown)
{

    STDMETHOD(put_EnableTextOnVideo) (THIS_
    				  BOOL bEnableTextOnVideo       /* [in] */	// Is Text On Video Enabled
				 ) PURE;

    STDMETHOD(get_EnableTextOnVideo) (THIS_
    				  BOOL *pbEnableTextOnVideo      /* [out] */	
				 ) PURE;

    STDMETHOD(put_PreDefRadio) (THIS_
    				  BOOL bPreDefRadio       /* [in] */	// Is pre-defined position selected ?
				 ) PURE;

    STDMETHOD(get_PreDefRadio) (THIS_
    				  BOOL *pbPreDefRadio      /* [out] */	//
				 ) PURE;

    STDMETHOD(put_PreDefPosition) (THIS_
    				  DWORD dwPreDefPosition       /* [in] */	// 0 Top, 1 Middle, 2 Bottom 
				 ) PURE;

    STDMETHOD(get_PreDefPosition) (THIS_
    				  DWORD *pdwPreDefPosition      /* [out] */	//
				 ) PURE;

    STDMETHOD(put_StartX) (THIS_
    				  DWORD dwStartX       /* [in] */	// 
				 ) PURE;

    STDMETHOD(get_StartX) (THIS_
    				  DWORD *pdwStartX      /* [out] */	//
				 ) PURE;

    STDMETHOD(put_StartY) (THIS_
    				  DWORD dwStartY       /* [in] */	// 
				 ) PURE;

    STDMETHOD(get_StartY) (THIS_
    				  DWORD *pdwStartY      /* [out] */	//
				 ) PURE;

    STDMETHOD(put_EndX) (THIS_
    				  DWORD dwEndX       /* [in] */	// 
				 ) PURE;

    STDMETHOD(get_EndX) (THIS_
    				  DWORD *pdwEndX      /* [out] */	//
				 ) PURE;

    STDMETHOD(put_EndY) (THIS_
    				  DWORD dwEndY       /* [in] */	// 
				 ) PURE;

    STDMETHOD(get_EndY) (THIS_
    				  DWORD *pdwEndY      /* [out] */	//
				 ) PURE;

    STDMETHOD(put_DisplayDate) (THIS_
    				  BOOL bDisplayDate       /* [in] */	// Display Date on the video
				 ) PURE;

    STDMETHOD(get_DisplayDate) (THIS_
    				  BOOL *pbDisplayDate      /* [out] */	
				 ) PURE;

    STDMETHOD(put_DisplayTime) (THIS_
    				  BOOL bDisplayTime       /* [in] */	// Display Time on the video ?
				 ) PURE;

    STDMETHOD(get_DisplayTime) (THIS_
    				  BOOL *pbDisplayTime      /* [out] */	//
				 ) PURE;

    STDMETHOD(put_DisplayLocation) (THIS_
    				  BOOL bDisplayLocation       /* [in] */	// Display Location on the video ?
				 ) PURE;

    STDMETHOD(get_DisplayLocation) (THIS_
    				  BOOL *pbDisplayLocation      /* [out] */	//
				 ) PURE;

    STDMETHOD(put_ReplaceLocation) (THIS_
    				  BOOL bReplaceLocation       /* [in] */	// Replace the location with user input text
				 ) PURE;

    STDMETHOD(get_ReplaceLocation) (THIS_
    				  BOOL *pbReplaceLocation      /* [out] */	//
				 ) PURE;

    STDMETHOD(put_ReplaceText) (THIS_
    				  BSTR szText       /* [in] */	// Display Time on the video ?
				 ) PURE;

    STDMETHOD(get_ReplaceText) (THIS_
    				  BSTR *pszText      /* [out] */	//
				 ) PURE;

    STDMETHOD(UpdateData) (THIS_
    				  BOOL bUpdate       /* [in] */	// Replace the location with user input text
				 ) PURE;
};
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
// IFontProperty
//----------------------------------------------------------------------------
DECLARE_INTERFACE_(IFontProperty, IUnknown)
{
    STDMETHOD(put_FaceName) (THIS_
    				  BSTR pszFaceName       /* [in] */	// Font name
				 ) PURE;

    STDMETHOD(get_FaceName) (THIS_
    				  BSTR *ppszFaceName      /* [out] */	
				 ) PURE;

    STDMETHOD(put_FontSize) (THIS_
    				  int nSize       /* [in] */
				 ) PURE;

    STDMETHOD(get_FontSize) (THIS_
    				  int *pnSize      /* [out] */	//
				 ) PURE;

    STDMETHOD(put_Italic) (THIS_
    				  BOOL bItalic       /* [in] */	
				 ) PURE;

    STDMETHOD(get_Italic) (THIS_
    				  BOOL *pbItalic      /* [out] */	//
				 ) PURE;

    STDMETHOD(put_UnderLine) (THIS_
    				  BOOL bUnderLn       /* [in] */	// 
				 ) PURE;

    STDMETHOD(get_UnderLine) (THIS_
    				  BOOL *pbUnderLn      /* [out] */	//
				 ) PURE;

    STDMETHOD(put_Weight) (THIS_
    				  LONG lWeight       /* [in] */	// 
				 ) PURE;

    STDMETHOD(get_Weight) (THIS_
    				  LONG *plWeight      /* [out] */	//
				 ) PURE;
    STDMETHOD(put_Strikeout) (THIS_
    				  BOOL bStrikeout       /* [in] */	
				 ) PURE;

    STDMETHOD(get_Strikeout) (THIS_
    				  BOOL *pbStrikeout      /* [out] */	//
				 ) PURE;

    STDMETHOD(put_TextColor) (THIS_
    				  DWORD dwColor       /* [in] */	// 
				 ) PURE;

    STDMETHOD(get_TextColor) (THIS_
    				  DWORD * pdwColor      /* [out] */	//
				 ) PURE;

    STDMETHOD(put_BkColor) (THIS_
    				  DWORD dwColor       /* [in] */	// 
				 ) PURE;

    STDMETHOD(get_BkColor) (THIS_
    				  DWORD * pdwColor      /* [out] */	//
				 ) PURE;

    STDMETHOD(put_CharSet) (THIS_
    				  BYTE byCharSet       /* [in] */	// 
				 ) PURE;

    STDMETHOD(get_CharSet) (THIS_
    				  BYTE *pbyCharSet      /* [out] */	//
				 ) PURE;

    STDMETHOD(put_Transparent) (THIS_
    				  BOOL bTransparent       /* [in] */	
				 ) PURE;

    STDMETHOD(get_Transparent) (THIS_
    				  BOOL *pbTransparent      /* [out] */	//
				 ) PURE;

	STDMETHOD(Apply) (THIS_ ) PURE;
};

#ifdef __cplusplus
}
#endif

#endif // __ITEXTONVIDEO__
