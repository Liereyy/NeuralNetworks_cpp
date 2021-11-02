#include <iostream>

int f1(int* y);
int f2(int x);

int main()
{
	int (*f[30])(int* y);
	int (*(*(*pg())(int x))[20])(int* y);
	int (*(*p)[20])(int* y);

	// int (dg())[20];
	// int* (*p)();
	// pg = f[20];
}