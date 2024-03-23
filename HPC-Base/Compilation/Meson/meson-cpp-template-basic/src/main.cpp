#include <iostream>
#include "mymath.hpp"
#include "sqrt.hpp"

int main(int argc, char** argv)
{
	std::cout << "Hello World!" << std::endl;
	std::cout << "2 + 2 = " << add(2, 2) << std::endl;
	std::cout << "2 - 2 = " << subtract(2, 2) << std::endl;
	std::cout << "2 * 2 = " << multiply(2, 2) << std::endl;
	std::cout << "2 / 2 = " << divide(2, 2) << std::endl;
	std::cout << "sqrt(4) = " << sqrt(4) << std::endl;
	std::cout << "sqrt(48,566,961) = " << sqrt(48566961) << std::endl;
}