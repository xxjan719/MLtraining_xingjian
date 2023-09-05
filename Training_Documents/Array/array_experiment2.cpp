
#include <Eigen/Dense>
#include <iostream>

using namespace Eigen;
using namespace std;

int main()
{
ArrayXXf a(3,3);
ArrayXXf b(3,3);
a << 1,2,3,
     4,5,6,
     7,8,9;

b<< 1,2,3,
    4,5,6,
    1,2,3;

// Adding two arrays
cout<< "a+b = " << endl << a+b << endl <<endl;

// Subtracting a scalar from an array
cout<< "a-2=" << endl << a-2 << endl;

ArrayXf c = ArrayXf:: Random(5);
cout<<"c = "<<endl<< c <<endl;
c*= 2;
cout<<"c = "<<endl<< c <<endl;
cout<< "c.abs() = "<< endl << c.abs() << endl;
cout<< "c.abs().sqrt() = "<< endl << c.abs().sqrt() << endl;
cout << "c.min(c.abs().sqrt())="<<endl << c.min(c.abs().sqrt())<<endl;

}

