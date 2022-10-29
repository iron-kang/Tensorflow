## tflite_gpu
This is a tflite gpu example
### Build tflite library
* X86_64
sudo apt install ocl-icd-opencl-dev
```shell=
bazel build -c opt //tensorflow/lite:libtensorflowlite.so
```
