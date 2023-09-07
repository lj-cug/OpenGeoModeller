#include "Registar.h"

Registrar::Registrar(std::string name, factoryMethod a_classFactoryFunction)
{
    // register the class factory function 
    FileFormatFavorFactory::Instance()->RegisterFactoryFunction(name, a_classFactoryFunction);
}