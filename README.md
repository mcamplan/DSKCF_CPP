-----------------------------------------------------------------
#DS-KCF: A real-time tracker for RGB-D data#
-----------------------------------------------------------------

##Introduction
This is an open source implementation of **"DS-KCF: A real-time tracker for RGB-D data"** [1]. 
The code provide a real time C++ implementation of the DS-KCF RGBD tracker as presented in [1]
The code provided guarantees an average processing throughput of more than 180 frame per second

It is free for research use. If you find it useful, please acknowledge the paper reported below. 

[1] S. Hannuna, M. Camplani, J. Hall, M. Mirmehdi, D. Damen, T. Burghardt, A. Paiement, L. Tao, 
DS-KCF: A real-time tracker for RGB-D data, Journal of Real-Time Image Processing
http://dx.doi.org/10.1007/s11554-016-0654-3


##Build

This project can be compiled and used in both in Windows Linux operating system enviroments.
This project has been tested in Windows 7 (VS2013 and VS2010) and Linux (Ubuntu 14.04) with g++.

Use CMAKE or CMAKEGUI to properly configure your system by using the  CMakeLists.txt on the top directory

### Dependencies
* C++11
* OpenCV 3.0
* TBB
* SSE2-capable CPU

### Windows 7
* Set environment variables for TBB and OPENCV (see OpenCV Setup - Environment Variables http://docs.opencv.org/doc/tutorials/introduction/windows_install/windows_install.html#windowssetpathandenviromentvariable)
* Launch cmake-gui, create a build folder and configure.
* Open CfTracking.sln in Visual Studio and compile the projects DSKCFcpp.

### Ubuntu 14.04
* Install OpenCV 3.0, TBB, and CMake.
* Configure and compile:
```
mkdir <src-dir>/build
cd <src-dir>/build
cmake ../
make -j 8


## Usage: 

   DSKCFcpp  [--hog_linear] [--hog_concatenate] [--hog_depth]
               [--hog_colour] [--raw_linear] [--raw_concatenate]
               [--raw_colour] [--raw_depth] [-d]
               [--depth_image_name_expansion <string>] [--depth_sequence
               <path>] [--depth_cam <integer>] [--depth_mock_sequence]
               [--mock_sequence] [--start_frame <integer>] [-r] [-p] [-e
               <folder path>] [-o <file path>] [-n] [-b <x,y,w,h>] [-i
               <string>] [-s <path>] [-c <integer>] [--] [--version] [-h]


## Where:
   -d,  --depth
     Use an additional depth input

   --depth_image_name_expansion <string>
     depth image name expansion (only necessary for image sequences) ie.
     /%.05d.jpg

   --depth_sequence <path>
     Path to depth sequence

   --depth_cam <integer>
     Depth device id

   --depth_mock_sequence
     Instead of processing a regular sequence, a dummy sequence is used to
     evaluate run time performance.

   --mock_sequence
     Instead of processing a regular sequence, a dummy sequence is used to
     evaluate run time performance.

   --start_frame <integer>
     starting frame idx (starting at 1 for the first frame)

   -r,  --repeat
     endless loop the same sequence

   -p,  --paused
     Start paused

   -e <folder path>,  --export <folder path>
     Path to output folder where the images will be saved with BB

   -o <file path>,  --out <file path>
     Path to output file

   -n,  --no-show
     Don't show video

   -b <x,y,w,h>,  --box <x,y,w,h>
     Init Bounding Box

   -i <string>,  --image_name_expansion <string>
     image name expansion (only necessary for image sequences) ie.
     /%.05d.jpg

   -s <path>,  --seq <path>
     Path to sequence

   -c <integer>,  --cam <integer>
     Camera device id

   --,  --ignore_rest
     Ignores the rest of the labeled arguments following this flag.

   --version
     Displays version information and exits.

   -h,  --help
     Displays usage information and exits.

##Data Format.
Some issue can arise with particular naming convention of your data. Please look at the example data used in this repository at data.bris
to test the DSKCF matlab version

http://dx.doi.org/10.5523/bris.16vbnj3im1ygi1sh0yd0mt4lp0
	 
##Third Party Code and Dependencies
This project depends on 

* TBB https://www.threadingbuildingblocks.org/
* Piotr's Matlab Toolbox http://vision.ucsd.edu/~pdollar/toolbox/doc/
  and specifically the HOG feature descriptor presented in [3]
* OpenCV http://opencv.org/
* tclap http://tclap.sourceforge.net/

DS-KCF tracker has been inspired by KCF tracker proposed by Henriques et al in [2]

Part of the core KCF tracking code has been inspired from the C++ KCF implementation provided in
Haag, K.: KCF implementation C??. GitHub repository. https://github.com/klahaag/cf_tracking (2015) 
According to the original git repository mentioned above
The code using linear correlation filters may be affected by a US patent. If you want to use this code commercially in the US 
please refer to http://www.cs.colostate.edu/~vision/ocof_toolset_2012/index.php for possible patent claims. 

TBB is released under APACHE 2.0 License
OPENCV is released under (BSD License)

##References
[1] S. Hannuna, M. Camplani, J. Hall, M. Mirmehdi, D. Damen, T. Burghardt, A. Paiement, L. Tao, 
DS-KCF: A real-time tracker for RGB-D data, Journal of Real-Time Image Processing
http://dx.doi.org/10.1007/s11554-016-0654-3

[2] J. F. Henriques, R. Caseiro, P. Martins, and J. Batista. High-speed tracking with kernelized 
correlation filters. Pattern Analysis and Machine Intelligence, IEEE Transactions on, 2015.

[3] P. F. Felzenszwalb, R. B. Girshick, D. McAllester, D. Ramanan, 
Object Detection with Discriminatively Trained Part Based Models, 
IEEE Transactions on Pattern Analysis and Machine Intelligence 32 (9) (2010) 1627–1645

##License
This code is licensed under the BSD license, which means you can modify and use for any purpose. 
But, the third party libraries have different licenses. See the third party libraries section for more details
	 
Copyright (c) 2016, Jake Hall, Massimo Camplani, Sion Hannuna 
All rights reserved.
 
  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions
  are met:
  1. Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.
  2. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
 
  THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
  ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
  OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
  HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
  OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
  SUCH DAMAGE.