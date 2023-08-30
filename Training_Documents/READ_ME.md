# Some code tips for using git or github
This is project to upload the details by using git and github.
## Some regular code for Ubuntu Linux system
1. [Simple C++ coding](https://blog.csdn.net/w464960660/article/details/129357160)

```
sudo apt install vim //reading and downloading package
mkdir 01_hello_world
cd 01_hello_world
ls 01_hello_world //check other folder in this '01_hello_world' folder
vim 01_hello_world.cpp
```
Then we can edit in this '01_hello_world.cpp' file like this
```
#include<iostream>
using namespace std;
int main(){
cout<<"Hello World!"<<endl;
return 0;}
```
Then Run C++ for this coding:
```
g++ 01_hello_world.cpp -o 01_hello world // use g++ to run the code. 01_hello_world after -o is an output result.
./01_hello_world // Execute under the terminal, print "Hello, World!" and wrap.
```