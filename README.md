# optimized-ml-compiler

## Purpose of this Repo

The purpose of this repo is to really understand how model run. In university I learnt how to build models and how to evaluate models but I have no idea how a computer will actually run a model. This repo is setup to create a model compuiler with the intention of (1) learning how running a model actually works and (2) ways to make this runtime compiler more efficient. 

## ML Model Compiler Design 



### Example ONNX File 

This repo will utilize a number of example ONNX ML model files to test it's infrastructure. The first ONNX file has version 1.1 of AlexNet a CNN model for image classification. Unfortunately the file is 250MB, which although is small it's too large to commit to GitHub. 


## Setup 

This project depends on ONNX and by extension Protobuf (and others). In order to run Opty you'll have to configure these dependencies on your machine. The following guide should take you through whats necessary for a smooth build process. 

### Building ONNX
Fortunately ONNX configures Protobuf for us so you'll only need to worry about configuring ONNX. This repo comes with an empty directory for it's dependencies "./third_party". You'll need to populate this directory using the following commands:

```bash
cd third_party
git submodule add https://github.com/onnx/onnx.git
cd onnx
git submodule update --init --recursive
```

This command will configure ONNX within the third_party directory. Now you'll be able to build the project with the following commands (assuming you're still in the onnx directory):

```bash
cd ../..
mkdir build && cd build
cmake ..
make -j$(nproc)
```

Then you can run the program via:

```bash
./opty
```