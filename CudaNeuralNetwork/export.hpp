#ifndef __EXPORT__
#define __EXPORT__

#ifdef BUILD_DLL
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT __declspec(dllimport)
#endif

#endif // !__EXPORT__
