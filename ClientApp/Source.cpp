#include <cpg/cpg_std_extensions.hpp>

#include "CudaMath.hpp"

int main()
{
	int a[]{ 1, 2, 3, 4, 5 };
	int b[]{ 5, 4, 3, 2, 1 };
	int c[5]{};

	std::cout << "Before: " << std::endl;
	std::cout << "a = " << a << std::endl;
	std::cout << "b = " << b << std::endl;
	std::cout << "c = " << c << std::endl;
	 
	add(c, a, b, 5);

	std::cout << "\nAfter: " << std::endl;
	std::cout << "a = " << a << std::endl;
	std::cout << "b = " << b << std::endl;
	std::cout << "c = " << c << std::endl;
}