#ifndef __VEC_DLL__
#define __VEC_DLL__

struct Vec2
{
	float x;
	float y;
};

struct Vec3
{
	double x;
	double y;
	double z;

	Vec3(double xi, double yi, double zi)
	{
		x = xi;
		y = yi;
		z = zi;
	}
};

#endif//!__VEC_DLL__