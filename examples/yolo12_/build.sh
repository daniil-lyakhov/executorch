rm -r build
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH="/home/dlyakhov/Projects/executorch_ov/executorch/cmake-out/;;home/dlyakhov/Projects/ultralytics/examples/YOLOv8-LibTorch-CPP-Inference/libtorch"  ..
make -j 30
