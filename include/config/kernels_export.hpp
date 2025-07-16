#ifdef _WIN32
  #ifdef KERNEL_EXPORTS
    #define KERNEL_API extern "C" __declspec(dllexport)
  #else
    #define KERNEL_API extern "C"
  #endif
#else
  #define KERNEL_API extern "C"
#endif