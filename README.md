# CUDA Raymarching
This app demonstrates a raymarching program in CUDA. It uses bazel build system.

## Dependencies
You will need to install bazel for your respective system. Instructions can be found [here.](https://bazel.build/install)
Additionally you will need CUDA, NVCC, and g++. On Arch linux I just run
```bash sudo pacman -S cuda nvidia``` but this is system dependent. You will of course need an Nvidia GPU as well.

## Instructions
To build and run, do: 
```bash
bazel build //src:main
bazel run //src:main
```
