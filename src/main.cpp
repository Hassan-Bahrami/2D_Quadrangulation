#include <iostream>
#include <utility>
#include <vector>
#include <random>
#include <Eigen/Dense>
#include <igl/readOBJ.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiPlugin.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>


struct Triangle {
    Eigen::RowVector3d normal;
    Eigen::RowVector3d v1, v2, v3; // Vertices of the face
};

// Define a structure to store a grid point's position, index, and adjacent vertices
struct Point {
    Eigen::RowVector3d position;
    int index;
    std::vector<int> neighbors;
    Eigen::RowVector3d baryCoord;
    int triangleIdx;
};

// Define the Quad structure
struct Quad {
    int vertices[4]; // Store the vertex indices of the quad
    Eigen::RowVector2d minPoint;
    Eigen::RowVector2d maxPoint;

    // Function to get the actual vertex positions from the grid vertices
    Eigen::RowVector3d getVertexPosition(const std::vector<Point>& gridVertices, int index) const {
        return gridVertices[vertices[index]].position;
    }
};

// Define a structure to represent a mesh
struct Mesh {
    std::vector<Point> vertices;
    std::vector<Quad> faces;
    Mesh(const std::vector<Point>& vertices, const std::vector<Quad>& faces)
        : vertices(vertices), faces(faces) {}
};


// Function to determine which axis is effectively zero in a mesh
int FindZeroAxis(const Eigen::MatrixXd& vertices) {
    int zeroAxis;
    // Define a threshold for considering an axis as effectively zero
    const double epsilon = 1e-6;
    // Check the x-axis
    bool isXZero = true;
    double xValue = vertices(0, 0);
    for (int i = 1; i < vertices.rows(); ++i) {
        if (std::abs(vertices(i, 0) - xValue) > epsilon) {
            isXZero = false;
            break;
        }
    }

    // Check the y-axis
    bool isYZero = true;
    double yValue = vertices(0, 1);
    for (int i = 1; i < vertices.rows(); ++i) {
        if (std::abs(vertices(i, 1) - yValue) > epsilon) {
            isYZero = false;
            break;
        }
    }

    // Check the z-axis
    bool isZZero = true;
    double zValue = vertices(0, 2);
    for (int i = 1; i < vertices.rows(); ++i) {
        if (std::abs(vertices(i, 2) - zValue) > epsilon) {
            isZZero = false;
            break;
        }
    }

    // Determine which axis is effectively zero
    if (isXZero) zeroAxis = 0; // X-axis is effectively zero
    else if (isYZero) zeroAxis = 1; // Y-axis is effectively zero
    else if (isZZero) zeroAxis = 2; // Z-axis is effectively zero
    else zeroAxis = -1; // None of the axes is effectively zero

    return zeroAxis;
}

// Function to calculate the 2D bounding box of the mesh based on the zero axis
void CalculateBoundingBox(const Eigen::MatrixXd& vertices, int zeroAxis, Eigen::RowVector2d& minPoint, Eigen::RowVector2d& maxPoint) {
    // Determine the indices of the non-zero columns
    int col1 = (zeroAxis + 1) % 3;
    int col2 = (zeroAxis + 2) % 3;

    for (int i = 0; i < vertices.rows(); ++i) {
        minPoint[0] = std::min(minPoint[0], vertices(i, col1));
        maxPoint[0] = std::max(maxPoint[0], vertices(i, col1));

        minPoint[1] = std::min(minPoint[1], vertices(i, col2));
        maxPoint[1] = std::max(maxPoint[1], vertices(i, col2));
    }
    // Increase min and max values by 20%
    double increaseFactor = 1.2;
    minPoint *= increaseFactor;
    maxPoint *= increaseFactor;
}

// Function to create a uniform 2D grid within the specified bounding box along the non-zero axis
std::vector<Eigen::RowVector2d> CreateUniformGrid2D(
    const Eigen::RowVector2d& minPoint,
    const Eigen::RowVector2d& maxPoint,
    double cellSize
) {
    std::vector<Eigen::RowVector2d> grid2D;

    for (double x = minPoint[0]; x <= maxPoint[0]; x += cellSize) {
        for (double y = minPoint[1]; y <= maxPoint[1]; y += cellSize) {
            grid2D.push_back(Eigen::RowVector2d(x, y));
        }
    }

    return grid2D;
}

std::vector<Point> CreateUniformGrid3DWithNeighbors(
    const Eigen::RowVector2d& minPoint,
    const Eigen::RowVector2d& maxPoint,
    double cellSize,
    int zeroAxis
) {
    std::vector<Point> grid3D;

    // Determine the indices of the non-zero axes
    int axis1 = (zeroAxis + 1) % 3;
    int axis2 = (zeroAxis + 2) % 3;

    int numRows = static_cast<int>((maxPoint[1] - minPoint[1]) / cellSize) + 1;
    int numColumns = static_cast<int>((maxPoint[0] - minPoint[0]) / cellSize) + 1;

    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numColumns; ++j) {
            double x = minPoint[0] + j * cellSize;
            double y = minPoint[1] + i * cellSize;

            Eigen::RowVector3d point;
            point[axis1] = x;
            point[axis2] = y;
            point[zeroAxis] = 0;

            Point gridPoint;
            gridPoint.position = point;
            gridPoint.index = i * numColumns + j;
            grid3D.push_back(gridPoint);
        }
    }

    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numColumns; ++j) {    
            // Check for x-wise neighbors to the left (left)
            Point gridPoint = grid3D[i*numColumns+j];
            // Check for x-wise neighbors to the left (left)
            if (j > 0) {
                grid3D[i * numColumns + j - 1].neighbors.push_back(gridPoint.index);
                gridPoint.neighbors.push_back(gridPoint.index - 1);
            }

            // Check for x-wise neighbors to the right (right)
            if (j < numColumns - 1) {
                grid3D[i * numColumns + j + 1].neighbors.push_back(gridPoint.index);
                gridPoint.neighbors.push_back(gridPoint.index + 1);
            }

            // Check for y-wise neighbors above (top)
            if (i > 0) {
                grid3D[(i - 1) * numColumns + j].neighbors.push_back(gridPoint.index);
                gridPoint.neighbors.push_back(gridPoint.index - numColumns);
            }

            // Check for y-wise neighbors below (bottom)
            if (i < numRows - 1) {
                grid3D[(i + 1) * numColumns + j].neighbors.push_back(gridPoint.index);
                gridPoint.neighbors.push_back(gridPoint.index + numColumns);
            }
        }
    }

    return grid3D;
}

// Function to derive quads from the  grid points
std::vector<Quad> DeriveQuadsFromGrid(const Eigen::RowVector2d& minPoint,
    const Eigen::RowVector2d& maxPoint,
    double cellSize,std::vector<Point> grid3D
){
    // Create a vector to store the quads
    std::vector<Quad> quads;

    int numRows = static_cast<int>((maxPoint[1] - minPoint[1]) / cellSize) + 1;
    int numColumns = static_cast<int>((maxPoint[0] - minPoint[0]) / cellSize) + 1;

    for (int i = 0; i < numRows - 1; ++i) {
        for (int j = 0; j < numColumns - 1; ++j) {
            // Get the current point and its neighbors
            Point& current = grid3D[i * numColumns + j];
            Point& right = grid3D[i * numColumns + (j + 1)];
            Point& below = grid3D[(i + 1) * numColumns + j];
            Point& rightBelow = grid3D[(i + 1) * numColumns + (j + 1)];

            // Check if neighbors exist
            if (right.neighbors.size() > 0 && below.neighbors.size() > 0 && rightBelow.neighbors.size() > 0) {
                // Create a quad using the current, right, rightBelow, and below vertices
                Quad quad;
                quad.vertices[0] = current.index;
                quad.vertices[1] = below.index;      
                quad.vertices[2] = rightBelow.index;
                quad.vertices[3] = right.index;

                // Add the quad to the vector of quads
                quads.push_back(quad);
            }
        }
    }   

    return quads;
}


void MakeQuadMesh(
    const std::vector<Quad>& quads,
    const std::vector<Point>& grid3D,
    Eigen::MatrixXd& V,
    Eigen::MatrixXi& F) {
    // Create a mapping from grid point indices to V matrix indices
    std::unordered_map<int, int> gridToVMappings;
    int nextVMappedIndex = 0;

    // Calculate the number of quads
    int numQuads = static_cast<int>(quads.size());

    // Loop over quads to determine the number of unique vertices
    int numQuadPoints = 0;
    for (int i = 0; i < numQuads; ++i) {
        for (int j = 0; j < 4; ++j) {
            int gridPointIndex = quads[i].vertices[j];

            // Check if the grid point index has been mapped to a V matrix index
            if (gridToVMappings.find(gridPointIndex) == gridToVMappings.end()) {
                // Map the grid point index to the next available V matrix index
                gridToVMappings[gridPointIndex] = nextVMappedIndex;
                nextVMappedIndex++;
                numQuadPoints++; // Increment the number of unique quad points
            }
        }
    }

    // Initialize the V matrix with zeros and F with the correct size
    V = Eigen::MatrixXd::Zero(numQuadPoints, 3);
    F.resize(numQuads, 4);

    // Loop over quads to populate the V and F matrices
    int quadIndex = 0;
    for (int i = 0; i < numQuads; ++i) {
        for (int j = 0; j < 4; ++j) {
            int gridPointIndex = quads[i].vertices[j];
            int vmappedIndex = gridToVMappings[gridPointIndex];

            // Set the V matrix based on the mapped indices
            V.row(vmappedIndex) = grid3D[gridPointIndex].position;
            F(i, j) = vmappedIndex;
        }
    }
}


// Define a struct to hold both P1 and P2 matrices and quad edges
struct QuadEdges {
    Eigen::MatrixXd P1;
    Eigen::MatrixXd P2;
    std::vector<Quad> quadEdges; // Add this member to store quad edges
};

// Function to create edge points from the quad mesh
QuadEdges create_quadEdges(const std::vector<Quad>& quadMesh, const std::vector<Point>& grid3D) {
    Eigen::MatrixXd P1, P2;

    // Initialize a vector to store quad edges
    std::vector<Quad> quadEdges;

    // Loop over all quads and connect vertices counterclockwise
    for (int i = 0; i < quadMesh.size(); ++i) {
        const Quad& quad = quadMesh[i];
        for (int j = 0; j < 4; ++j) {
            int vrtIdx1 = quad.vertices[j];
            int vrtIdx2 = quad.vertices[(j + 1) % 4]; // Wrap around for the last edge
            P1.conservativeResize(P1.rows() + 1, 3);
            P2.conservativeResize(P2.rows() + 1, 3);
            P1.row(P1.rows() - 1) = grid3D[vrtIdx1].position;
            P2.row(P2.rows() - 1) = grid3D[vrtIdx2].position;
        }

        // Store the quad in the quadEdges member
        quadEdges.push_back(quad);
    }

    // Create a QuadEdges struct to return both matrices and quad edges
    QuadEdges result;
    result.P1 = P1;
    result.P2 = P2;
    result.quadEdges = quadEdges;
    std::vector<bool> isInside; // Indicates if each edge is inside

    return result;
}


void BaryCentricCoord(Point& point, const Triangle& triangle) {
    Eigen::RowVector3d a = triangle.v1;
    Eigen::RowVector3d b = triangle.v2;
    Eigen::RowVector3d c = triangle.v3;
    Eigen::RowVector3d v0 = b - a;
    Eigen::RowVector3d v1 = c - a;
    Eigen::RowVector3d v2 = point.position - a;

    float d00 = v0.dot(v0);
    float d01 = v0.dot(v1);
    float d11 = v1.dot(v1);
    float d20 = v2.dot(v0);
    float d21 = v2.dot(v1);

    float denom = d00 * d11 - d01 * d01;

    // Check if the denominator is zero (or close to zero)
    if (std::abs(denom) < 1e-10) {
        std::cerr << "Denominator is too small: " << denom << std::endl;
        std::cerr << "Triangle vertices: " << a << ", " << b << ", " << c << std::endl;
        std::cerr << "Point position: " << point.position << std::endl;
        return;
    }

    Eigen::RowVector3d baryCoord;
    baryCoord.y() = (d11 * d20 - d01 * d21) / denom;
    baryCoord.z() = (d00 * d21 - d01 * d20) / denom;
    baryCoord.x() = 1.0f - baryCoord.y() - baryCoord.z();

    point.baryCoord = baryCoord;

    // Debug output to verify calculations
    std::cout << "Barycentric Coordinates: " << baryCoord << std::endl;
    std::cout << "Reconstructed Position: " << (baryCoord.x() * a + baryCoord.y() * b + baryCoord.z() * c) << std::endl;
}


// Function to check if a point is inside a triangle
bool isPointInsideTriangle(const Point& point, const Triangle& triangle) {
    // Calculate vectors and barycentric coordinates as before
    Eigen::RowVector3d e0 = triangle.v2 - triangle.v1;
    Eigen::RowVector3d e1 = triangle.v3 - triangle.v1;
    Eigen::RowVector3d e2 = point.position - triangle.v1;

    // Calculate the dot products
    double dot00 = e0.dot(e0);
    double dot01 = e0.dot(e1);
    double dot02 = e0.dot(e2);
    double dot11 = e1.dot(e1);
    double dot12 = e1.dot(e2);

    // Calculate barycentric coordinates
    double denom = dot00 * dot11 - dot01 * dot01;
    double u = (dot11 * dot02 - dot01 * dot12) / denom;
    double v = (dot00 * dot12 - dot01 * dot02) / denom;

    return (u >= 0 && v >= 0 && u + v <= 1);
}

// Function to compute barycentric coordinates and store the triangle for each point
void QuadsBarycentricCoords(
    Point& point, const std::vector<Triangle>& triangles) {
    int numIntersections = 0;
    int closestTriangleIndex = -1;
    double minDistanceSquared = std::numeric_limits<double>::max();

    for (int i = 0; i < triangles.size(); ++i) {
        const Triangle& triangle = triangles[i];

        if (isPointInsideTriangle(point, triangle)) {
            numIntersections++;
            BaryCentricCoord(point, triangle); // Correctly compute barycentric coordinates
            // No need to store in barycentric_quads; directly stored in point.baryCoord
            point.triangleIdx = i;
        } else {
            // If the point is not inside the triangle, check if it's the closest
            Eigen::RowVector3d pointToVertexDistance = point.position - triangle.v1;
            double distanceSquared = pointToVertexDistance.squaredNorm();
            
            if (distanceSquared < minDistanceSquared) {
                minDistanceSquared = distanceSquared;
                closestTriangleIndex = i;
            }
        }
    }

    if (numIntersections == 0 && closestTriangleIndex >= 0) {
        // If the point is not inside any triangle but has a closest triangle
        const Triangle& closestTriangle = triangles[closestTriangleIndex];
        BaryCentricCoord(point, closestTriangle); // Correctly compute barycentric coordinates
        // No need to store in barycentric_quads; directly stored in point.baryCoord
        point.triangleIdx = closestTriangleIndex;
    }
}

// Check all quad points if they are inside of the orginal mesh
bool isPointInsideMesh(const Point& point, const std::vector<Triangle>& triangles) {
    int numIntersections = 0;

    for (const Triangle& triangle : triangles) {
        // Calculate the vectors from the point to each vertex of the triangle
        Eigen::RowVector3d e0 = triangle.v2 - triangle.v1;
        Eigen::RowVector3d e1 = triangle.v3 - triangle.v1;
        Eigen::RowVector3d e2 = point.position - triangle.v1;

        // Calculate the dot products
        double dot00 = e0.dot(e0);
        double dot01 = e0.dot(e1);
        double dot02 = e0.dot(e2);
        double dot11 = e1.dot(e1);
        double dot12 = e1.dot(e2);

        // Calculate barycentric coordinates
        double denom = dot00 * dot11 - dot01 * dot01;
        double u = (dot11 * dot02 - dot01 * dot12) / denom;
        double v = (dot00 * dot12 - dot01 * dot02) / denom;

        // Check if the point is inside the triangle
        if (u >= 0 && v >= 0 && u + v <= 1) {
            numIntersections++;
        }
    }

    return numIntersections % 2 != 0; // Inside if odd number of intersections
}

// Convert Faces (F) from libigl to Triangles
std::vector<Triangle> FaceTotriangles(Eigen::MatrixXd V, Eigen::MatrixXi F)
{
    // Convert the faces to triangles
    std::vector<Triangle> triangles;
    for (int i = 0; i < F.rows(); ++i) {
        Triangle triangle;
        triangle.v1 = V.row(F(i, 0));
        triangle.v2 = V.row(F(i, 1));
        triangle.v3 = V.row(F(i, 2));
        triangles.push_back(triangle);
    }

    return triangles;
}

// Separating inside and outside edges of the quads in the mesh
QuadEdges separateQuadEdges(
    const std::vector<Quad>& quadMesh,
    const std::vector<Point>& grid3D_with_Neighbours,
    const std::vector<Triangle>& triangles
) {
    QuadEdges insideEdges, outsideEdges;

    // Create a vector to keep track of which vertices are inside the mesh
    std::vector<bool> isVertexInside(grid3D_with_Neighbours.size(), false);

    // Mark vertices that are inside the mesh
    for (int i = 0; i < grid3D_with_Neighbours.size(); ++i) {
        if (isPointInsideMesh(grid3D_with_Neighbours[i], triangles)) {
            isVertexInside[i] = true;
        }
    }

    // Loop through all quad edges in quadMesh
    for (const Quad& quad : quadMesh) {
        bool isInsideEdge = true; // Assume the edge is inside until proven otherwise

        // Check each vertex of the quad
        for (int i = 0; i < 4; ++i) {
            int vrtIdx1 = quad.vertices[i];

            // Check if the vertex is not inside the mesh
            if (!isVertexInside[vrtIdx1]) {
                isInsideEdge = false; // Mark as outside edge
                break; // No need to check other vertices
            }
        }

        // Separate the edge based on whether it's inside or outside
        if (isInsideEdge) {
            // Populate P1 and P2 with the edge coordinates
            for (int i = 0; i < 4; ++i) {
                int vrtIdx1 = quad.vertices[i];
                int vrtIdx2 = quad.vertices[(i + 1) % 4]; // Wrap around for the last edge
                insideEdges.P1.conservativeResize(insideEdges.P1.rows() + 1, 3);
                insideEdges.P2.conservativeResize(insideEdges.P2.rows() + 1, 3);
                insideEdges.P1.row(insideEdges.P1.rows() - 1) = grid3D_with_Neighbours[vrtIdx1].position;
                insideEdges.P2.row(insideEdges.P2.rows() - 1) = grid3D_with_Neighbours[vrtIdx2].position;
            }

            insideEdges.quadEdges.push_back(quad);
        } else {
            outsideEdges.quadEdges.push_back(quad);
        }
    }

   return insideEdges; // Return the separated inside edges
}

QuadEdges addOutsideBorderToInsideEdges(
    const std::vector<Quad>& quadMesh,
    const std::vector<Point>& grid3D_with_Neighbours,
    const std::vector<Triangle>& triangles,
    const std::vector<bool>& isOutsideBorder
) {
    // Create a separate list for inside and outside border vertices
    std::vector<int> insideVertices;
    std::vector<int> outsideBorderVertices;

    for (int i = 0; i < grid3D_with_Neighbours.size(); ++i) {
        if (isPointInsideMesh(grid3D_with_Neighbours[i], triangles)) {
            insideVertices.push_back(i);
        } else if (isOutsideBorder[i]) {
            outsideBorderVertices.push_back(i);
        }
    }

    // Create a new list of quads formed by inside vertices and outside border vertices
    std::vector<Quad> newQuads;

    for (const Quad& quad : quadMesh) {
        bool isQuadValid = true;

        for (int i = 0; i < 4; ++i) {
            int vrtIdx = quad.vertices[i];

            // Check if the vertex is not inside and not in the outside border
            if (!isPointInsideMesh(grid3D_with_Neighbours[vrtIdx], triangles) &&
                std::find(outsideBorderVertices.begin(), outsideBorderVertices.end(), vrtIdx) == outsideBorderVertices.end()) {
                isQuadValid = false;
                break;
            }
        }

        if (isQuadValid) {
            newQuads.push_back(quad);
        }
    }

    // Derive quad edges from the filtered quads
    QuadEdges result = create_quadEdges(newQuads, grid3D_with_Neighbours);

    return result;
}

// Function to calculate the bounding box (min and max points) for a quad
auto computeBoundingBox = [](const Quad& quad, const std::vector<Point>& gridVertices) -> std::pair<Eigen::RowVector3d, Eigen::RowVector3d> {
    Eigen::RowVector3d minPoint = quad.getVertexPosition(gridVertices, 0);
    Eigen::RowVector3d maxPoint = quad.getVertexPosition(gridVertices, 0);

    for (int i = 1; i < 4; ++i) {
        // int vertexIndex = quad.vertices[i];
        int vertexIndex = i;
        minPoint = minPoint.cwiseMin(quad.getVertexPosition(gridVertices, vertexIndex));
        maxPoint = maxPoint.cwiseMax(quad.getVertexPosition(gridVertices, vertexIndex));
    }

    return { minPoint, maxPoint };
};

// Function to find the quad that contains a point (search function)
int findQuadForPoint(const Eigen::RowVector3d& point, const std::vector<Quad>& quads, const std::vector<Point>& gridVertices) {
    for (int i = 0; i < quads.size(); ++i) {
        const Quad& quad = quads[i];
        auto [minPoint, maxPoint] = computeBoundingBox(quad, gridVertices);

        // Check if the point is inside the quad by testing if it's inside the bounding box
        if (point[0] >= minPoint[0] && point[0] <= maxPoint[0] &&
            point[1] >= minPoint[1] && point[1] <= maxPoint[1] &&
            point[2] >= minPoint[2] && point[2] <= maxPoint[2]) {
            return i;  // Return the index of the quad
        }
    }
    // If the point is not inside any quad, return -1 or some other indicator of no quad found.
    return -1;
}

// Function to convert F_quad to a vector of Quad objects
std::vector<Quad> FMatToFQuads(const Eigen::MatrixXi& F_quad) {
    std::vector<Quad> F_quads;

    int numQuads = F_quad.rows();
    for (int i = 0; i < numQuads; ++i) {
        Quad quad;
        for (int j = 0; j < 4; ++j) {
            quad.vertices[j] = F_quad(i, j);
        }
        F_quads.push_back(quad);
    }
    return F_quads;
}

std::vector<Quad> FilterQuadsInsideMesh(
    const std::vector<Quad>& quads,
    const std::vector<Triangle>& triangles,
    const std::vector<Point>& grid3D) {
    std::vector<Quad> quadsInsideMesh;

    for (const Quad& quad : quads) {
        bool isQuadInside = false;

        for (int i = 0; i < 4; ++i) {
            int vertexIndex = quad.vertices[i];
            Point point = grid3D[vertexIndex];

            if (isPointInsideMesh(point, triangles)) {
                isQuadInside = true;
                break;  // No need to check other vertices if one is inside
            }
        }

        if (isQuadInside) {
            quadsInsideMesh.push_back(quad);
        }
    }

    return quadsInsideMesh;
}

// Function to generate random values for the zero axis of the original mesh
Eigen::MatrixXd GenerateRandomZeroAxis(const Eigen::MatrixXd& V, int zeroAxis) {
    // Use a random number generator to create random values for the zero-axis.
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);

    Eigen::MatrixXd V_3D = V;

    // Replace the zero-axis values with random values to create curvature.
    for (int i = 0; i < V.rows(); ++i) {
        V_3D(i, zeroAxis) = distribution(gen);
    }

    return V_3D;
}

// Function to convert Eigen::MatrixXd to std::vector<Point>
std::vector<Point> EigenMatToVector(const Eigen::MatrixXd& mat) {
    std::vector<Point> points;
    
    for (int i = 0; i < mat.rows(); ++i) {
        Point point;
        point.position = mat.row(i);
        // You can set other Point members here if needed
        points.push_back(point);
    }
    
    return points;
}

int countPointsWithNegativeBarycentric(const std::vector<Eigen::RowVector3d>& barycentric_quads) {
    int count = 0;
    for (const Eigen::RowVector3d& baryCoords : barycentric_quads) {
        if (baryCoords.minCoeff() < 0.0) {
            count++;
        }
    }
    return count;
}


void updateTriangles(std::vector<Triangle>& triangles, const Eigen::MatrixXd& V_3D_Mat, const Eigen::MatrixXi& F)
{
    // Update triangle vertices based on the updated V_3D matrix
    for (int i = 0; i < triangles.size(); ++i) {
        triangles[i].v1 = V_3D_Mat.row(F(i, 0));
        triangles[i].v2 = V_3D_Mat.row(F(i, 1));
        triangles[i].v3 = V_3D_Mat.row(F(i, 2));
    }
}


int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_obj_file.obj>" << std::endl;
        return 1;
    }

    Eigen::MatrixXd V; // Vertices
    Eigen::MatrixXi F; // Faces

    // Load the OBJ mesh using the path provided in the command line
    if (!igl::readOBJ(argv[1], V, F)) {
        std::cerr << "Error loading OBJ file: " << argv[1] << std::endl;
        return 1;
    }
    
    // Find which axis is effectively zero
    int zeroAxis = FindZeroAxis(V);
    std::cout << "Axis " << zeroAxis << " is effectively zero." << std::endl;
    if (zeroAxis == -1) {
        std::cerr << "No axis is effectively zero." << std::endl;
        return 1;
    }

    Eigen::RowVector2d minPoint, maxPoint;
    // Calculate the 2D bounding box based on the zero axis
    CalculateBoundingBox(V, zeroAxis, minPoint, maxPoint);
    std::cout << "Min Point: " << minPoint << std::endl;
    std::cout << "Max Point: " << maxPoint << std::endl;

    // Define the cell size for the uniform grid
    double cellSize = 1.0;

    // Create a uniform 3D grid within the bounding box along the non-zero axis
    std::vector<Point> grid3D_with_Neighbours = CreateUniformGrid3DWithNeighbors(minPoint, maxPoint, cellSize, zeroAxis);
    // Convert grid3DPositions to MatrixXd
    Eigen::MatrixXd gridVertices(grid3D_with_Neighbours.size(), 3);
    for (size_t i = 0; i < grid3D_with_Neighbours.size(); ++i) {
        gridVertices.row(i) = grid3D_with_Neighbours[i].position;
    }

    // Set the color for the grid points (e.g., green)
    Eigen::MatrixXd gridColors(gridVertices.rows(), 3);
    gridColors.setZero();
    gridColors.col(1).array() = 1.0; // Set green channel to 1

    // // Define the index of the selected point (replace with your desired index)
    // int selectedPointIndex = 40; // Replace with the index of the point you want to select
    // // Highlight the selected point in red
    // gridColors.row(selectedPointIndex) << 1.0, 0.0, 0.0; // Red color
    // // Get the neighbors of the selected point (replace with your logic)
    // const std::vector<int>& neighbors = grid3D_with_Neighbours[selectedPointIndex].neighbors;
    // // Highlight the neighbors in blue
    // for (int neighbor : neighbors) {
    //     gridColors.row(neighbor) << 0.0, 0.0, 1.0; // Blue color
    // }

    // Derive quads from grid points
    std::vector<Quad> F_Quads = DeriveQuadsFromGrid(minPoint, maxPoint, cellSize, grid3D_with_Neighbours);

    // Make quad mesh
    Eigen::MatrixXd V_Quad_Mat;
    Eigen::MatrixXi F_Quad_Mat;
    MakeQuadMesh(F_Quads, grid3D_with_Neighbours, V_Quad_Mat, F_Quad_Mat);

    // Derive quad edges to be able to plot them
    QuadEdges quadEdges = create_quadEdges(F_Quads, grid3D_with_Neighbours);

    // Convert libgil V and F to vector of Triangles
    std::vector<Triangle> triangles = FaceTotriangles(V, F);

    // Shrink quad mesh to those quads that have overlap with original mesh
    std::vector<Quad> F_FitQuads = FilterQuadsInsideMesh(F_Quads, triangles, grid3D_with_Neighbours);

    // Make quad Mesh of fit quads
    Eigen::MatrixXd V_FitQuad_Mat;
    Eigen::MatrixXi F_FitQuad_Mat;
    MakeQuadMesh(F_FitQuads, grid3D_with_Neighbours, V_FitQuad_Mat, F_FitQuad_Mat);
    std::cout << "V_FitQuads: " << V_FitQuad_Mat.rows() << std::endl;
    std::cout << "F_FitQuad_Mat: " << F_FitQuad_Mat.rows() << std::endl;

    // Convert V_FitMatQuad to std::vector<Point>
    std::vector<Point> V_Quad = EigenMatToVector(V_Quad_Mat);
    std::vector<Point> V_FitQuad = EigenMatToVector(V_FitQuad_Mat);

    // Derive quad edges to be able to plot them
    QuadEdges FitQuadEdges = create_quadEdges(F_FitQuads, grid3D_with_Neighbours);

    // // *** Check barycentric coordinates in 3D *** //
    // // Initialize a vector to store barycentric coordinates of filtered quads
    // std::vector<Eigen::RowVector3d> barycentric_quads;
    // // Create a set to keep track of processed vertex indices
    // std::set<int> processedVertices;
    // // Compute barycentric coordinates for each filtered quad
    // for (const Quad& quad : F_FitQuads) {
    //     for (int i = 0; i < 4; ++i) {
    //         int vertexIndex = quad.vertices[i];
    //         if (processedVertices.find(vertexIndex) == processedVertices.end()) {
    //             Point point = grid3D_with_Neighbours[vertexIndex];
    //             QuadsBarycentricCoords(point, triangles);
    //             processedVertices.insert(vertexIndex);
    //         }
    //     }
    // }
    // // Assign random values to the zero axis of V to create a double curvature 3D mesh.
    // Eigen::MatrixXd V_3D_Mat = GenerateRandomZeroAxis(V, zeroAxis);
    // // update the trianlge values using new V_3D_Mat
    // updateTriangles(triangles, V_3D_Mat, F);
    // // Convert V_FitMatQuad to std::vector<Point>
    // std::vector<Point> V_3D = EigenMatToVector(V_3D_Mat);

    // // Define a vector to store the positions of quad points in 3D
    // std::vector<Point> V_quad3D;
    // // Create a set to keep track of processed vertex indices
    // std::set<int> processedVertices2;
    // // Loop over all quads and compute barycentric coordinates
    // for (const Quad& quad : F_FitQuads) {
    //     for (int i = 0; i < 4; ++i) {
    //         int vertexIndex = quad.vertices[i];

    //         // Check if the vertex index has already been processed
    //         if (processedVertices2.find(vertexIndex) == processedVertices2.end()) {
    //             Point point = grid3D_with_Neighbours[vertexIndex];
    //             const int triangleIndex = point.triangleIdx;

    //             // Check if triangleIndex is valid
    //             if (triangleIndex < 0 || triangleIndex >= triangles.size()) {
    //                 std::cerr << "Invalid triangle index: " << triangleIndex << std::endl;
    //                 continue;
    //             }

    //             const Triangle& triangle = triangles[triangleIndex];  // Access triangles directly

    //             // Print debug information
    //             std::cout << "Processing Point Index: " << point.index << std::endl;
    //             std::cout << "Triangle Index: " << triangleIndex << std::endl;
    //             std::cout << "Triangle vertices: " << triangle.v1 << ", " << triangle.v2 << ", " << triangle.v3 << std::endl;

    //             // Check vector lengths
    //             Eigen::RowVector3d v0 = triangle.v2 - triangle.v1;
    //             Eigen::RowVector3d v1 = triangle.v3 - triangle.v1;
    //             Eigen::RowVector3d v2 = point.position - triangle.v1;

    //             if (v0.norm() < 1e-10 || v1.norm() < 1e-10 || v2.norm() < 1e-10) {
    //                 std::cerr << "Zero length vector detected." << std::endl;
    //                 continue;
    //             }

    //             // Compute barycentric coordinates
    //             BaryCentricCoord(point, triangle);

    //             // Reconstruct the 3D position using barycentric coordinates
    //             Eigen::RowVector3d reconstructed = point.baryCoord.x() * triangle.v1 +
    //                                             point.baryCoord.y() * triangle.v2 +
    //                                             point.baryCoord.z() * triangle.v3;

    //             // Check if reconstructed coordinates match actual coordinates
    //             if (!reconstructed.isApprox(point.position, 1e-6)) {
    //                 std::cout << "!!! Wrong reconstructed coordinate - Point index: " << point.index << " has different reconstructed and actual coordinates." << std::endl;
    //             }

    //             // Create a Point structure and store the computed 3D position
    //             Point quadPoint;
    //             quadPoint.position = reconstructed;
    //             V_quad3D.push_back(quadPoint);

    //             processedVertices2.insert(vertexIndex);
    //         } 
    //     }
    // }


    // std::cout << "V_quad3D: " << V_quad3D.size() << std::endl;
    // std::cout << "F_FitQuads: " << F_FitQuads.size() << std::endl;
    // // // Derive quad edges in 3D to be able to plot them
    // QuadEdges Quad3DEdges = create_quadEdges(F_FitQuads, V_quad3D);


    // int negativeBary = countPointsWithNegativeBarycentric(barycentric_quads);
    // std::cout << "Negative bary points: " << negativeBary << std::endl;
    // std::cout << "barycentric_quads: " << barycentric_quads.size() << std::endl;
    // std::cout << "V_quads_Mat: " << V_Quad_Mat.rows() << std::endl;
    // std::cout << "F_Quad_Mat: " << F_Quad_Mat.rows() << std::endl;
    // std::cout << "Grid points: " << grid3D_with_Neighbours.size() << std::endl;
    // std::cout << "V_FitQuads: " << V_FitQuad.size() << std::endl;
    // std::cout << "F_FitQuads: " << F_FitQuads.size() << std::endl;
    // std::cout << "V_FitQuads_Mat: " << V_FitQuad_Mat.rows() << std::endl;
    // std::cout << "F_FitQuad_Mat: " << F_FitQuad_Mat.rows() << std::endl;
    // std::cout << "V_3D_Mat: " << V_3D_Mat.rows() << std::endl;

    // // *** End of checking barycentric coordinates *** //

    for (size_t i=0; i<grid3D_with_Neighbours.size(); i++)
    {
        Point quadPoint = grid3D_with_Neighbours[i];

        // Check if the point is inside of the Original mesh
        bool isInside = isPointInsideMesh(quadPoint, triangles);

        if (isInside)
        {
            gridColors.row(i) << 1.0, 0.0, 0.0;
        }
    }

    // Initialize the isOutsideBorder vector with all points as outside
    std::vector<bool> isOutsideBorder(grid3D_with_Neighbours.size(), false);

    // Loop through all vertices in grid3D_with_Neighbours
    for (int i = 0; i < grid3D_with_Neighbours.size(); ++i) {
        Point& vertex = grid3D_with_Neighbours[i];
        
        // Check if this vertex is inside the mesh
        if (!isPointInsideMesh(vertex, triangles)) {
            // Loop through the neighbors of this vertex
            for (int neighborIdx : vertex.neighbors) {
                // Check if the neighbor is inside of the mesh
                if (isPointInsideMesh(grid3D_with_Neighbours[neighborIdx], triangles)) {
                    isOutsideBorder[i] = true; // Mark as outside border
                    break; // No need to check other neighbors
                }
            }
        }
    }

    // Highlight the outside border vertices (vertices marked as isOutsideBorder)
    for (int i = 0; i < grid3D_with_Neighbours.size(); ++i) {
        if (isOutsideBorder[i]) {
            gridColors.row(i) << 0.0, 0.0, 1.0; // Blue color
        }
    }

    // Visualize inside edges with one color
    QuadEdges insideEdges = separateQuadEdges(F_Quads, grid3D_with_Neighbours, triangles);
    QuadEdges combinedEdges = addOutsideBorderToInsideEdges(F_Quads, grid3D_with_Neighbours, triangles, isOutsideBorder);

     // Initialize the libigl viewer
    igl::opengl::glfw::Viewer viewer;

    // Create an ImGui plugin
    igl::opengl::glfw::imgui::ImGuiPlugin plugin;
    viewer.plugins.push_back(&plugin);
    igl::opengl::glfw::imgui::ImGuiMenu menu;
    plugin.widgets.push_back(&menu);

    // Variables to control visibility
    bool showOriginalMesh = false;
    bool showGridVertices = false;
    bool showGridQuads = false;
    bool showInsideEdges = false;
    bool showCombinedEdges = false;
    bool ShowFitQuadEdges = false;
    bool Show3DQuadEdges = false;
    bool ClearScene = false;
    bool showMesh = false;
    bool Show3DMesh = false;
    bool Vert1_bary = false;
    bool Vert1_act = false;
    bool ShowTriangle = false;

    // Customize the menu
    double doubleVariable = 0.1f; // Shared between two menus

    // // Callback to create the custom menu
    // Add content to the default menu window
    menu.callback_draw_viewer_menu = [&]()
    {
        // Draw parent menu content
        menu.draw_viewer_menu();

        // Add new group
        if (ImGui::CollapsingHeader("Quadrangulation", ImGuiTreeNodeFlags_DefaultOpen))
        {

            if (ImGui::Checkbox("Show Original Mesh", &showMesh))
            {
                viewer.data().set_mesh(V, F);
            }

            if (ImGui::Checkbox("Show Gird points", &showGridVertices))
            {
                viewer.data().add_points(gridVertices, gridColors);
            }

            if (ImGui::Checkbox("Show Gird quads", &showGridQuads))
            {
                viewer.data().add_edges(quadEdges.P1, quadEdges.P2, Eigen::RowVector3d(1.0, 0.0, 0.0));
            }

            if (ImGui::Checkbox("Regular quads fit inside", &showInsideEdges))
            {
                viewer.data().add_edges(insideEdges.P1, insideEdges.P2, Eigen::RowVector3d(0.0, 1.0, 0.0));
            }

            // if (ImGui::Checkbox("Regular quads contains whole mesh", &showCombinedEdges))
            // {
            //     viewer.data().add_edges(combinedEdges.P1, combinedEdges.P2, Eigen::RowVector3d(0.0, 0.0, 1.0));
            // }

            if (ImGui::Checkbox("Regular quads contains whole mesh", &ShowFitQuadEdges))
            {
                viewer.data().add_edges(FitQuadEdges.P1, FitQuadEdges.P2, Eigen::RowVector3d(0.0, 0.0, 1.0));
            }


            // if (ImGui::Checkbox("3D quads contains whole mesh", &Show3DQuadEdges))
            // {
            //     viewer.data().add_edges(Quad3DEdges.P1, Quad3DEdges.P2, Eigen::RowVector3d(0.0, 0.0, 1.0));
            // }

            // if (ImGui::Checkbox("Add 3D mesh", &Show3DMesh))
            // {
            //     viewer.data().set_mesh(V_3D_Mat, F);
            // }

            if (ImGui::Button("Clear scene"))
            {
                viewer.data().clear();
            }


            // // Compute the coordinates of each filtered quad using their barycentric coordinates
            // const Quad& quad = F_FitQuads[50];
            // Eigen::RowVector3d quadPoint3D;
            // int vertexIdx1 = quad.vertices[2];
            // Point point1 = grid3D_with_Neighbours[vertexIdx1];
            // const int triangleIndex = point1.triangleIdx;
            // const Triangle& triangle = triangles[triangleIndex];  // Access triangles directly
            // Eigen::RowVector3d baryCoords = computeBarycentricCoordinates(point1, triangle);
            // Interpolate the 3D position using barycentric coordinates
            // Eigen::RowVector3d quadPoint3D_point1 = baryCoords(0) * triangle.v1 +
            //                                         baryCoords(1) * triangle.v2 +
            //                                         baryCoords(2) * triangle.v3;

            // BaryCentricCoord(point1, triangle);
            //  Eigen::RowVector3d recontructed = point1.baryCoord.x() * triangle.v1 +
            //                                          point1.baryCoord.y() * triangle.v2 +
            //                                          point1.baryCoord.z() * triangle.v3;
            // std::cout << "Barycentric Coordinate: " << point1.baryCoord << std::endl;
            // std::cout << "Actual Coordinate: " << point1.position << std::endl;
            // std::cout << "Reconstructed Coordinate: " << recontructed << std::endl;
            // Eigen::MatrixXd point1_bary = Eigen::MatrixXd::Zero(1, 3);
            // point1_bary.row(0) = recontructed;
            // Eigen::MatrixXd point1_act = Eigen::MatrixXd::Zero(1, 3);
            // point1_act.row(0) = point1.position;
            // int vertex2 = quad.vertices[1];
            // int vertex3 = quad.vertices[2];
            // int vertex4 = quad.vertices[3];


        //     if (ImGui::Checkbox("Vertex 1 bary", &Vert1_bary))
        //     {   
        //         viewer.data().add_points(point1_bary, Eigen::RowVector3d(1.0, 0.0, 0.0));
        //     }

        //     if (ImGui::Checkbox("Vertex 1 actual", &Vert1_act))
        //     {
        //         viewer.data().add_points(point1_act, Eigen::RowVector3d(0.0, 1.0, 0.0));
        //     }

        //     if (ImGui::Checkbox("Show Triangle", &ShowTriangle))
        //     {
        //         viewer.data().add_points(triangle.v1, Eigen::RowVector3d(0.0, 0.0, 1.0));
        //         viewer.data().add_points(triangle.v2, Eigen::RowVector3d(0.0, 0.0, 1.0));
        //         viewer.data().add_points(triangle.v3, Eigen::RowVector3d(0.0, 0.0, 1.0));
        //     }            

        }
    };

    // *** visulaize mapped original mesh and fit quad mesh in 3D *** //
    // Add 3D original mesh to the scene
    viewer.data().set_mesh(V, F);


    // // Add 3D quad edges to the scene
    // viewer.data().add_edges(Quad3DEdges.P1, Quad3DEdges.P2, Eigen::RowVector3d(1.0, 0.0, 0.0));

    // Customize the viewer as needed
    // viewer.data().set_mesh(V, F);
    // Launch the viewer
    viewer.launch();

    return 0;
};