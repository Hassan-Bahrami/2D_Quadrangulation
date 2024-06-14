# 2D Quadrangulation
This is a C++ project to fit a Quad grid on a 2D mesh.
The fitted grid is either can be completely covered by the mesh (inside of the mesh) or completely can covere the 2D mesh (The border quads have overlaps with the outside as well).

## Dependencies
The code relies on:
- glfw
- glad
- imGui

All of them are already included in the label. So, you can simply use "libigl".

## How to use
To use the code, you need to use the following commands from the root directory of the project:

```bash
mkdir build
cd build
cmake ..
make
```
After building the project, go to the build directory and run:

```bash
2D_Quadrangulation <path to the 2D mesh>
```

## Usage example
After building the project, go to the build directory and run:

```bash
2D_Quadrangulation ../models/curvedPlane.obj
```
You should see a glfw window open and show a 2D plane. Then, you can fit the quad grids on it using the "Quadrangulation" section of the GUI.
Here is a preview of the project:




