
#include "FileFormatFavorFactory.h"


 FileFormatFavorInterface*  FileFormatFavorFactory::CreateInstance(std::string a_name)
 {
	 std::map<std::string, factoryMethod>::iterator registeredPair =
     FileFormatFavorFactory::m_factoryFunctionRegistry.find(a_name);
    // did we find one?
    if(registeredPair == FileFormatFavorFactory::m_factoryFunctionRegistry.end())
     return NULL; // return NULL
   // return a new instance of derived class
    return (registeredPair->second());
 }
 
 bool  FileFormatFavorFactory::RegisterFactoryFunction(std::string name,factoryMethod a_classFactoryFunction)
 {
	 // add the pair to the map
     std::pair<std::map<std::string, factoryMethod>::iterator, bool> registeredPair =
      FileFormatFavorFactory::m_factoryFunctionRegistry.insert(std::make_pair(name.c_str(), a_classFactoryFunction));
     // return whether it was added or updated
      return registeredPair.second;
 }
 
 FileFormatFavorFactory *  FileFormatFavorFactory::Instance()
 {
	  static FileFormatFavorFactory factory;
      return &factory;
 }

