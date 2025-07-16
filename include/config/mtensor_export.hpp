
//to export the functions and clases to the .dll files 
#ifdef _WIN32
  #ifdef MTENSOR_EXPORTS
    #define MTENSOR_API __declspec(dllexport)
  #else
    #define MTENSOR_API __declspec(dllimport)
  #endif
#else
  #define MTENSOR_API
#endif