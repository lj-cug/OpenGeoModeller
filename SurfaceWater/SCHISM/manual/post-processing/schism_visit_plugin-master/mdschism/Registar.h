
#ifndef _REGISTRAR_H_
#define _REGISTRAR_H_  
#include "FileFormatFavorFactory.h"

class Registrar {
public:
    Registrar(std::string className, factoryMethod a_classFactoryFunction);
};

#define REGISTER_CLASS(NAME, TYPE) \
    static Registrar registrar(NAME, \
        [](void) ->  FileFormatFavorInterface*  { return new TYPE();});

#endif