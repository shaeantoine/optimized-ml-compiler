# Opty: An ONNX ML Model Compiler

## Purpose of this Repo

The purpose of this repo is to really understand how models run. In university I learnt how to build ML models and how to evaluate said models but I have no idea how a computer actually runs them. This repo is setup to create a model compuiler with the intention of (1) learning how running a model actually works and (2) ways to make this runtime compiler more efficient. 

## Repo Structure 

```bash
opty/
├── src/
│   ├── operations/
│   │    ├── add.cpp
│   │    ├── matmul.cpp
│   │    ├── relu.cpp
│   │    ├── sigmoid.cpp
│   │    ├── dropout.cpp
│   │    ├── concat.cpp
│   │    ├── conv.cpp
│   │    ├── maxpool.cpp
│   │    ├── softmax.cpp
│   │    ├── globalaveragepool.cpp
│   │    └── CMakeLists.txt
│   ├── CMakeLists.txt
│   ├── execution_context.cpp
│   ├── operator_registry.cpp
│   ├── tensor.cpp
│   ├── ir.cpp
│   ├── main.cpp                # Where the Model Graph is run from 
├── include/
│   ├── operations/
│   │    ├── add.hpp
│   │    ├── matmul.hpp
│   │    ├── relu.hpp
│   │    ├── dropout.hpp
│   │    ├── concat.hpp
│   │    ├── conv.hpp
│   │    ├── maxpool.hpp
│   │    ├── softmax.hpp
│   │    ├── globalaveragepool.hpp
│   │    └── sigmoid.hpp
│   ├── execution_context.hpp
│   ├── operator_registry.hpp
│   ├── onnx_to_ir.hpp
│   ├── operator.hpp
│   ├── tensor.hpp
│   └── ir.hpp
├── CMakeLists.txt
├── .gitmodules
├── .gitignore
├── LICENSE
├── SYSTEMDESIGN.md
└── README.md
```

### Example ONNX File 

This repo is compatible with any ONNX file. You may find any number of interesting ML ONNX models on [ONNX's Model Github](https://github.com/onnx/models). Currently the repo assumes a model is stored within the directory opty/model. For the purposes of development and testing I've used "squeezenet1.0-3.onnx" which you may find more information about [here](https://github.com/forresti/SqueezeNet).

## Setup 

This project depends on ONNX and by extension Protobuf (and others). In order to run Opty you'll have to configure these dependencies on your machine. The following guide should take you through whats necessary for a smooth build process. 

### Building ONNX
Fortunately ONNX configures Protobuf for us so you'll only need to worry about configuring ONNX. However, you WILL need to install protobuf as a global OS package. THe can be done with the following command using Homebrew on MacOS:

```bash
brew install protobuf
```

This repo comes with an empty directory for it's dependencies "./third_party". You'll need to populate this directory using the following commands:

```bash
mkdir third_party && cd third_party
git submodule add -f https://github.com/onnx/onnx.git
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
./src/opty
```