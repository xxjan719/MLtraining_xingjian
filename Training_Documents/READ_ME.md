# Some code tips for using git or github
This is project to upload the details by using git and github.

## [Ubuntu git download](https://blog.csdn.net/beishanyingluo/article/details/107451921)
```
sudo apt-get install git
git config --global user.name "aaa"
git config --global user.email "aa@example.com"
ssh-keygen -t rsa -C"aa@example.com"
```


## [Some os codes for using](https://blog.csdn.net/l_liangkk/article/details/78729059) 
I always forget some codes. This is for remembering them.
```
cd ..  //Return to the previous directory.
cd ../..  //Return to the previous second directory.
cd or cd ~ //Return to home page.
cd - content // Return to the specific directory.
```
## Some regular code for Ubuntu Linux system
1. [Simple C++ coding by using g++](https://blog.csdn.net/w464960660/article/details/129357160)

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
2. [Simple C++ coding by using Cmake](https://zhuanlan.zhihu.com/p/110513954)

   
About how to use it, please see the link. There only is the exploring code for using Cmake
```
cd Desktop
mkdir cppSpace
cd cppSpace
vim HelloWorld.cpp
```

click 'o' in keyboard to unlock the editing condition.

Then typing this following code
```
#include<iostream>
using namespace std;
int main(){
cout<<"Hello world!"<<endl;
return 0;}
```

Then click 'Esc' in keyboard to lock the editing condition.

Then click 'Alt' + ':' in keyboard to generate ':' in the screen.

Then click 'wq' to exit the cpp file(w means saving the file, q means editing the file.)

In folder cppSpace, we create .txt file named 'CMakeLists.txt'(This name is very important and can not be removed. If you do it, the code will generate debug.) And we open it to write this following down in the file.
```
cmake_minimum_required(VERSION 2.8)
project(HelloWorld)
add_executable(Helloworld HelloWorld.cpp) #(程序名 源代码文件)
```
Then we type this in terminal:
```
cmake .
make
```
Compared with directly using g++ instructions to compile each program and source file, when we use cmake to compile a C++ project, we only need to manage and maintain __a file called CMakeLists.txt__. For example: If we want to add another executable file, we only need to add a line of "add_executable" command in CMakeLists.txt, and the subsequent steps do not need to be changed.


When compiling according to the above process, the only regret is that the intermediate files generated during the compilation process stay in this folder (cppSpace folder). These intermediate files need to be removed when we release the code, which causes some inconvenience. , the solution is: Create a new intermediate directory (folder) to store these intermediate files, and delete this intermediate directory directly after the compilation is successful. So the more common way is in the terminal:
```
mkdir build
cd build
cmake ..
make
```
3. [Multiple source files in the same directory in Cmake](https://blog.csdn.net/whahu1989/article/details/82078563)

Have already done! We need to pay more attention to the following structure section.

4. [Structure for Cmake](https://blog.csdn.net/whahu1989/article/details/82078563)
```
.
.
├── bin
├── include
│   ├── testFunc1.h
│   └── testFunc.h
└── src
    ├── main.cpp
    ├── testFunc1.cpp
    └── testFunc.cpp
```
```
cd bin
cmake ..
make
./main
```
## [download arpack-ng](https://github.com/opencollab/arpack-ng)
```
$ git clone git@github.com:opencollab/arpack-ng.git
$ cd ./arpack-ng
$ which cmake && cmake --version
$ mkdir build
$ cd build
$ cmake -D EXAMPLES=ON -D MPI=ON -D BUILD_SHARED_LIBS=ON ..
$ make
$ sudo make install
```
## [download eigen]([https://github.com/opencollab/arpack-ng](https://blog.csdn.net/p942005405/article/details/100653731])
```
$ #include <iostream>
$ #include <Eigen/Dense>
 
$ //using Eigen::MatrixXd;
$ using namespace Eigen;
$ using namespace Eigen::internal;
$ using namespace Eigen::Architecture;
 
$ using namespace std;
 
$ int main()
$ {
$        cout<<"*******************1D-object****************"<<endl;
$        Vector4d v1;
$        v1<< 1,2,3,4;
$        cout<<"v1=\n"<<v1<<endl;
$ 
$        VectorXd v2(3);
$        v2<<1,2,3;
$        cout<<"v2=\n"<<v2<<endl;
$ 
$        Array4i v3;
$        v3<<1,2,3,4;
$        cout<<"v3=\n"<<v3<<endl;
$ 
$        ArrayXf v4(3);
$        v4<<1,2,3;
$        cout<<"v4=\n"<<v4<<endl;
$ }
```
```
$ g++ eigen-test.cpp -o eigen-test
```


