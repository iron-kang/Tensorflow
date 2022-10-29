# tflite_gpu
This is a tflite gpu example
## Build tflite library
### X86_64
* Requirement package
  1. OpenCL
  ```shell=
  sudo apt install ocl-icd-opencl-dev
  ```
  2. OpenCV
  ```shell=
  sudo apt-get install -y \
    libopencv-core-dev \
    libopencv-highgui-dev \
    libopencv-calib3d-dev \
    libopencv-features2d-dev \
    libopencv-imgproc-dev \
    libopencv-video-dev \
    libopencv-contrib-dev \
    libopencv-dev
    ```
* Build command
```shell=
git clone https://github.com/tensorflow/tensorflow.git tensorflow_src
bazel build -c opt //tensorflow/lite:libtensorflowlite.so
bazel build -s --copt="-DCL_TARGET_OPENCL_VERSION=300" tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_delegate.so
```
### AArch64/Arm
* Build command
```shell=
git clone https://github.com/tensorflow/tensorflow.git tensorflow_src
bazel build -s -c opt --config=elinux_aarch64(elinux_armhf) --copt="-DEGL_NO_X11" --copt="-DCL_TARGET_OPENCL_VERSION=220" tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_delegate.so
bazel build --config=elinux_aarch64(elinux_armhf) -c opt //tensorflow/lite:libtensorflowlite.so
```
