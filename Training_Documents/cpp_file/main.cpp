

#include <cmath> //mathematical using

#include <iostream>
#include <complex>
#include <vector> // vector has a class for arbitrary uses and the following template<typename T> will use these construction.
#include <arpack.hpp>
#include <debug_c.hpp>
#include <stat_c.hpp>
using namespace std;
//template <typename Real> // this declares a template for the function, which allows it to work with any data type specified by the user. The data type will be refered to as 'Real' inside the function. Typically, in this context, 'Real' would be some kind of floating-point data type,like 'float' or 'double', but it can technically be any type the user specifies when calling the function.
//void diagonal_matrix_vector_product(const Real* x, Real* y){ //void: meaning it doesn't return any value. It takes two parameters. const means that you can't modify the values in array 'x' within the function.
//for (int i=0; i<1000;++i){
//	y[i] = static_cast<Real>(i+1)*x[i];
//}
//} // the diagonal matrix is implicitly represented by the loop index 'i'. The value on the diagonal is 'i+1'.
  // In performance-sensitive scenarios, pre-increment(++i) can sometimes be slightly faster than post-increment (i++),especially when dealing with iterators in C++ because post-increment often involves creating a tempoaray object.
  // static_cast<Real>(i+1) is used to convert the integer value 'i+1' to the type 'Real'.

template <typename Real>
void diagonal_matrix_vector_product(const complex<Real>* x, complex<Real>* y){
for (int i=0;i<1000;++i){
//Use complex matrix (i,-i) instead of (i,i): this way "larget_magnitude"
//and "largest_imaginary" options produce different results that can be checked.

	y[i] = x[i]*complex<Real>{Real(i+1),-Real(i+1)};
}
}
int main()
{ // EX1:
  //float x[1000],y[1000];
  //Initialize x with some values, for the example,x[i] = i*0.5
  //for (int i=0;i<1000;++i){
 // x[i] = i*0.5f;
 // }
 // diagonal_matrix_vector_product(x,y);
 //
 //EX2:
  complex<double> x[2] = {{1,2},{2,3}};
  complex<double> y[2];
  diagonal_matrix_vector_product<double>(x,y);
  //print the result;
  for (int i=0;i<3;++i){
  cout<<"y["<<i<<"]="<<y[i]<<endl;
  }


  //EX1 print:
  //for (int i=0;i<10;++i){
//cout<< "x["<< i << "]="<<x[i]<<", y["<<i<<"]="<<y[i]<<endl;
 // }
   printf("Hello CMake\n");
   return 0;


}
