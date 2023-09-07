#ifndef _FILEFORMATFAVORFACTORY_H_
#define _FILEFORMATFAVORFACTORY_H_  
#include <string>
#include <map>
#include "FileFormatFavorInterface.h"


typedef FileFormatFavorInterface* (*factoryMethod)();

class FileFormatFavorFactory
{

public:

	  //return a derived file format favor impl class according to name
       FileFormatFavorInterface* CreateInstance(std::string a_name);
	   //register a derived impl class into factory
	   bool RegisterFactoryFunction(std::string name,factoryMethod a_classFactoryFunction);
	   //get factory singleton instance
       static FileFormatFavorFactory * Instance();
       virtual ~FileFormatFavorFactory() {};

private:

	  std::map<std::string, factoryMethod> m_factoryFunctionRegistry;
	  
};


#endif