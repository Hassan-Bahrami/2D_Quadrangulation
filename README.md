# 2D Quadrangulation
This is a C++ project to fit a Quad grid on a 2D mesh.
The fitted grid can either completely cover the mesh (inside) or extend beyond it (border quads overlap with the outside).

## Dependencies
The code relies on:
- glfw
- glad
- imGui

All of them are already included in the label. So, you can simply use "libigl".

## How to use
To build and run the project, follow these steps from the root directory:

```bash
mkdir build
cd build
cmake ..
make
```
After building the project, navigate to the build directory and execute:

```bash
2D_Quadrangulation <path to the 2D mesh>
```

## Usage example
Once built, navigate to the build directory and run:

```bash
2D_Quadrangulation ../models/curvedPlane.obj
```
This will open a glfw window displaying a 2D plane. You can fit quad grids onto it using the "Quadrangulation" section of the GUI.
Here is a preview of the project in action:


![Preview](./models/Preview.gif)

