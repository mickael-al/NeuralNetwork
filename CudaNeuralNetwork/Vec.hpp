//Peer Programming: Guo, Albarello
#ifndef __VEC_DLL__
#define __VEC_DLL__

struct Vec2
{
	float x;
	float y;
};

struct Vec3
{
	float x;
	float y;
	float z;

	Vec3(float xi, float yi, float zi)
	{
		x = xi;
		y = yi;
		z = zi;
	}
};

#endif//!__VEC_DLL__