
#include <eigen3/Eigen/Core>
#include<iostream>

// import most common Eigen types
// USING_PART_OF_NAMESPACE_EIGEN
using namespace Eigen;
using namespace std;
int main(int,char *[])
{
Matrix3f m3;
m3<<1,2,3,4,5,6,7,8,9;
Matrix4f m4 = Matrix4f::Identity();
Vector4i v4(1,2,3,4);

cout<< "m3\n"<<m3<<"\nm4:\n"
	<< m4 << "\nv4:\n" << v4 <<endl;
}


