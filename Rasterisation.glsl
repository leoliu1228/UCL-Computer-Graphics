#define PROJECTION
#define RASTERIZATION
#define CLIPPING
#define INTERPOLATION
#define ZBUFFERING
//#define ANIMATION

precision highp float;
uniform float time;

// Polygon / vertex functionality
const int MAX_VERTEX_COUNT = 8;

uniform ivec2 viewport;

struct Vertex {
vec3 position;
vec3 color;
};

struct Polygon {
// Numbers of vertices, i.e., points in the polygon
int vertexCount;
// The vertices themselves
Vertex vertices[MAX_VERTEX_COUNT];
};

// Appends a vertex to a polygon
void appendVertexToPolygon(inout Polygon polygon, Vertex element) {
for (int i = 0; i < MAX_VERTEX_COUNT; ++i) {
if (i == polygon.vertexCount) {
polygon.vertices[i] = element;
}
}
polygon.vertexCount++;
}

// Copy Polygon source to Polygon destination
void copyPolygon(inout Polygon destination, Polygon source) {
for (int i = 0; i < MAX_VERTEX_COUNT; ++i) {
destination.vertices[i] = source.vertices[i];
}
destination.vertexCount = source.vertexCount;
}

// Get the i-th vertex from a polygon, but when asking for the one behind the last, get the first again
Vertex getWrappedPolygonVertex(Polygon polygon, int index) {
if (index >= polygon.vertexCount) index -= polygon.vertexCount;
for (int i = 0; i < MAX_VERTEX_COUNT; ++i) {
if (i == index) return polygon.vertices[i];
}
}

// Creates an empty polygon
void makeEmptyPolygon(out Polygon polygon) {
polygon.vertexCount = 0;
}

// Clipping part

#define ENTERING 0
#define LEAVING 1
#define OUTSIDE 2
#define INSIDE 3

int getCrossType(Vertex poli1, Vertex poli2, Vertex wind1, Vertex wind2) {
#ifdef CLIPPING
// Put your code here
// The method is exactly same as half space testing done in edge() which uses cross product
// For better demonstration please refer to the edge() comments
// M = W1P x W1W2
// M.x =  W1P.y * W1W2.z −  W1P.z * W1W2.y <- Ignored
// M.y =  W1P.z * W1W2.x −  W1P.x * W1W2.z <- Ignored
// M.z =  W1P.x * W1W2.y −  W1P.y * W1W2.x <-> (poli1.position.x - wind1.position.x) * (wind2.position.y - wind1.position.y) - (poli1.position.y - wind1.position.y) * (wind2.position.x - wind1.position.x);

// Find value for each of the coordinates given
float resultPoli1 = (poli1.position.x - wind1.position.x) * (wind2.position.y - wind1.position.y) - (poli1.position.y - wind1.position.y) * (wind2.position.x - wind1.position.x);
float resultPoli2 = (poli2.position.x - wind1.position.x) * (wind2.position.y - wind1.position.y) - (poli2.position.y - wind1.position.y) * (wind2.position.x - wind1.position.x);

// Clipping the polygon against a boundaries
if(resultPoli1 < 0.0 && resultPoli2 > 0.0){
return ENTERING;
}
if(resultPoli1 > 0.0 && resultPoli2 < 0.0){
return LEAVING;
}
if(resultPoli1 < 0.0 && resultPoli2 < 0.0){
return OUTSIDE;
}
if(resultPoli1 > 0.0 && resultPoli2 > 0.0){
return INSIDE;
}
#else
return INSIDE;
#endif
}

// This function assumes that the segments are not parallel or collinear.
Vertex intersect2D(Vertex a, Vertex b, Vertex c, Vertex d) {
#ifdef CLIPPING
// Put your code here
// Calculate the point of intersection between two vertex and given boundaries
// In order to do so, we need to first derive two parametric equations from given vertex
// a, b are on line AB and c, d are on line CD
// By definition a line can be represented as ax + by + c = 0
// We will use capital letter to represent a, b, c, d for better demostration
// y - (B.y) = Slope * (x - B.x) where Slope = (B.y - A.y) / (B.x - A.x)
// Therefore:
// l(AB) = [(B.y - A.y) / (B.x - A.x)] * (x - B.x) + B.y - y
// l(CD) = [(D.y - C.y) / (D.x - C.x)] * (x - D.x) + D.y - y
// The intersected point can be found when l(AB) = l(CD)
// To find x value
// [(B.y - A.y) / (B.x - A.x)] * (x - B.x) + B.y - [(D.y - C.y) / (D.x - C.x)] * (x - D.x) + D.y = 0
// Let's say (B.y - A.y) / (B.x - A.x) = S1 and (D.y - C.y) / (D.x - C.x) = S2
// S1 * x - S1 * B.x + B.y - S2 * x + S2 * D.x - D.y = 0
// (S1 - S2)x = S1 * B.x - B.y - S2 * D.x + D.y
// x = (S1 * B.x - B.y - S2 * D.x + D.y) / (S1 - S2)
// Then we put the x back to the equation and can get y
// y = S1 * (x - B.x) + B.y;

// First we can simply derive the equations from using vectors:
vec3 A = a.position;
vec3 B = b.position;
vec3 C = c.position;
vec3 D = d.position;
float S1 = (B.y - A.y)/(B.x - A.x);
float S2 = (D.y - C.y)/(D.x - C.x);

Vertex intersectedPoint;

intersectedPoint.position.x = (S1 * B.x - B.y - S2 * D.x + D.y) / (S1 - S2);
intersectedPoint.position.y = S1 * (intersectedPoint.position.x - B.x) + B.y;

// Usually we could put the x, y back to the 3D equation and get z value
// However the interpolation is non-linear here, we need to use the perspective correct depth interpolation
float s = (intersectedPoint.position.x - A.x)/(B.x - A.x);

// The actual z value will be
// z = 1 / ((1 / Z1) + s * ((1 / Z2) - (1 / Z1)))
// = (Z1 * Z2) / (Z2 + s * (Z1 - Z2))
intersectedPoint.position.z = 1.0 / ((1.0 / A.z) + s * ((1.0 / B.z) - (1.0 / A.z)));
// = intersectedPoint.position.z = (A.z*B.z)/(B.z + s*(A.z - B.z));

// For color interpolation we need the value t which can be found using the following formula
// t = (s * Z1) / (s * Z1 + (1 - s) * Z2)
float t = s * A.z / (s * A.z + (1.0 - s)* B.z);
intersectedPoint.color = a.color + t *(b.color - a.color);

return intersectedPoint;
#else
return a;
#endif
}

void sutherlandHodgmanClip(Polygon unclipped, Polygon clipWindow, out Polygon result) {
Polygon clipped;
copyPolygon(clipped, unclipped);

// Loop over the clip window
for (int i = 0; i < MAX_VERTEX_COUNT; ++i) {
if (i >= clipWindow.vertexCount) break;

// Make a temporary copy of the current clipped polygon
Polygon oldClipped;
copyPolygon(oldClipped, clipped);

// Set the clipped polygon to be empty
makeEmptyPolygon(clipped);

// Loop over the current clipped polygon
for (int j = 0; j < MAX_VERTEX_COUNT; ++j) {
if (j >= oldClipped.vertexCount) break;

// Handle the j-th vertex of the clipped polygon. This should make use of the function
// intersect() to be implemented above.
#ifdef CLIPPING
// Put your code here
// Cycle through all vertex in clip region
Vertex wind1 = getWrappedPolygonVertex(clipWindow, i);
Vertex wind2 = getWrappedPolygonVertex(clipWindow, i+1);

// Cycle through all vertex in provided polygon
Vertex poli1 = getWrappedPolygonVertex(oldClipped, j);
Vertex poli2 = getWrappedPolygonVertex(oldClipped, j+1);

int crossType = getCrossType(poli1, poli2, wind1, wind2);

// Could use switch statement for better readability but it does not allow
// Else if statement has some bugs in certain browsers: exactly same code may produce totally different results (has been tested on UCL Ctrix and lab computers)
if (crossType == ENTERING){
// Enter the clip region, add poli2 and intersectede point
appendVertexToPolygon(clipped, intersect2D(poli1, poli2, wind1, wind2));
appendVertexToPolygon(clipped, poli2);
}
if(crossType == LEAVING){
// Leave the clip region, add only intersected point
appendVertexToPolygon(clipped, intersect2D(poli1, poli2, wind1, wind2));
}
if(crossType == INSIDE){
// Be entirely inside, add only poli2
appendVertexToPolygon(clipped, poli2);
}
if(crossType == OUTSIDE){
// One last possible type is OUTSIDE
// Be entirely outside, do nothing
}

#else
appendVertexToPolygon(clipped, getWrappedPolygonVertex(oldClipped, j));
#endif
}
}

// Copy the last version to the output
copyPolygon(result, clipped);
}

// Rasterization and culling part

#define INNER_SIDE 0
#define OUTER_SIDE 1

// Assuming a clockwise (vertex-wise) polygon, returns whether the input point
// is on the inner or outer side of the edge (ab)
int edge(vec2 point, Vertex a, Vertex b) {
#ifdef RASTERIZATION
// Put your code here
// We can simply check whether the point is on the inner or outer side by using the magnitude of cross product (this concept was just introduced in cgvi mathematics week6)
// By definition:
// U x V = |U||V|sinθ where the sign of the angle between the vectors determines the side
// Although cross product is usually applicable for 3D and 7D, we can derive the formula as 3D and ignore the z axis, e.g explicitly treating z as 0.0
// The result can be easily checked since |U||V| is always positive
// If it is greater than 0 then it means it is inside the projected plane -> sinθ is positive
// If it is smaller than 0 then it means it is outside the projected plane -> sinθ is negative

// Since we know point A, B and P, then we have vector AB, AP
vec3 AP = vec3(point.x, point.y, 0.0) - a.position;
vec3 AB = b.position-a.position;

// M = AP x AB
// By definition:
// M.x = AP.y * AB.z − AP.z * AB.y <- Ignored
// M.y = AP.z * AB.x − AP.x * AB.z <- Ignored
// M.z = AP.x * AB.y − AP.y * AB.x <-> (point.x - a.position.x) * (b.position.y - a.position.y) - (point.y - a.position.y) * (b.position.x - a.position.x)
// The equation does look like a dot product but it is truely a cross product for 2D matrix
float result = AP.x * AB.y - AP.y * AB.x;

if (result > 0.0){
return INNER_SIDE;
}

#endif
return OUTER_SIDE;
}

// Returns if a point is inside a polygon or not
bool isPointInPolygon(vec2 point, Polygon polygon) {
// Don't evaluate empty polygons
if (polygon.vertexCount == 0) return false;
// Check against each edge of the polygon
bool rasterise = true;
for (int i = 0; i < MAX_VERTEX_COUNT; ++i) {
if (i < polygon.vertexCount) {
#ifdef RASTERIZATION
// Put your code here
// Cycle through all the vertex to get the edge and determine whether the point is inside or outside the polygon
Vertex a = getWrappedPolygonVertex(polygon, i);
Vertex b = getWrappedPolygonVertex(polygon, i+1);
// If it is not inside the polygon we dismiss it otherwise rasterise it
int halfTest = edge(point,a,b);
if (halfTest != INNER_SIDE){
rasterise = false;
}
#else
rasterise = false;
#endif
}
}
return rasterise;
}

bool isPointOnPolygonVertex(vec2 point, Polygon polygon) {
for (int i = 0; i < MAX_VERTEX_COUNT; ++i) {
if (i < polygon.vertexCount) {
ivec2 pixelDifference = ivec2(abs(polygon.vertices[i].position.xy - point) * vec2(viewport));
int pointSize = viewport.x / 200;
if( pixelDifference.x <= pointSize && pixelDifference.y <= pointSize) {
return true;
}
}
}
return false;
}

float triangleArea(vec2 a, vec2 b, vec2 c) {
// https://en.wikipedia.org/wiki/Heron%27s_formula
float ab = length(a - b);
float bc = length(b - c);
float ca = length(c - a);
float s = (ab + bc + ca) / 2.0;
return sqrt(max(0.0, s * (s - ab) * (s - bc) * (s - ca)));
}

Vertex interpolateVertex(vec2 point, Polygon polygon) {
float weightSum = 0.0;
vec3 colorSum = vec3(0.0);
vec3 positionSum = vec3(0.0);
float depthSum = 0.0;
for (int i = 0; i < MAX_VERTEX_COUNT; ++i) {
if (i < polygon.vertexCount) {
#if defined(INTERPOLATION) || defined(ZBUFFERING)
// Put your code here
// Given a point and the target polygon, we need to construct a triangle for each loop to calculate the weight of the point
// Normally we can use i , i+1, i+2 to build a triangle, however, the getWrappedPolygonVertex() function only returns the first vertex
// e.g. when overflow happens, the i+1 ,i+2 will both give the first vertex
// Therefore we need to use i-1, i, i+1 and code the function to determine i-1 vertex

// Vertex A: i-1 -> The adjacent vertex 1
Vertex A;

if (i-1<0){
	// If overflow to negative index, we give it the last index in the vertex
	A = getWrappedPolygonVertex(polygon, polygon.vertexCount - 1);
}else{
	// Proceed as usual
	A = getWrappedPolygonVertex(polygon, i - 1);
}
vec2 a = vec2(A.position.x, A.position.y);

// Vertex B: i -> The target vertex
Vertex B = getWrappedPolygonVertex(polygon, i);
vec2 b = vec2(B.position.x, B.position.y);

// Vertex C: i+1 -> The adjacent vertex 2
Vertex C = getWrappedPolygonVertex(polygon, i+1);
vec2 c = vec2(C.position.x, C.position.y);

// According to Bicentric coordinates
// Weight of B = area of APC/ area of ABC
float weight = triangleArea(point,a,c)/triangleArea(a,b,c);
// Depth of B = z value of vertex B
float depth = B.position.z;

#else
#endif
#ifdef ZBUFFERING
// Put your code here
weightSum += weight/depth;
// Z adjusted
positionSum += B.position * (weight/depth);

#endif
#ifdef INTERPOLATION
// Put your code here
depthSum += weight/depth;
colorSum += B.color * (weight/depth);

#endif
}
}

Vertex result = polygon.vertices[0];

#ifdef INTERPOLATION
// Put your code here
result.color = colorSum/depthSum;
#endif
#ifdef ZBUFFERING
// Put your code here
result.position = positionSum/weightSum;
#endif
#if !defined(INTERPOLATION) && !defined(ZBUFFERING)
// Put your code here

#endif

return result;
}

// Projection part
// Used to generate a projection matrix.
mat4 computeProjectionMatrix() {
mat4 projectionMatrix = mat4(1);

float aspect = float(viewport.x) / float(viewport.y);
float imageDistance = 0.5;

#ifdef PROJECTION
// Put your code here
// Assume that our COP is initially located at (0,0,0)

mat4 translationUV, regularPyramid, scale, canonical = mat4(1);

// M(TranslationUV)	=
//  1   0   0   0
//  0   1   0   0
//  0   0   1   0
//  0   0   -d  1

float d = imageDistance;

translationUV[0] = vec4(1.0, 0, 0, 0);
translationUV[1] = vec4(0, 1.0, 0, 0);
translationUV[2] = vec4(0, 0, 1.0, -d);
translationUV[3] = vec4(0, 0, 0, 1.0);

// M(Regular Pyramid) =
//   2D/dx      0       0   0
//     0      2D/dy     0   0
//   -px/dx  -py/dy     1   0
// -(px/dx)D -(py/dy)D  0   1

// Where dx/dy => aspect

//	 |                                      v2   Top-Right: v2(1,y2)
//   |                                      |
//   |                   *                  |    * Center: TP
//   |                                      |
//   v1                                     |    Bottom-Left: v1(-1,y1)

// x(i) = U(i) - COP.x, i = 1, 2
// y(i) = V(i) - COP.y, i = 1, 2
// D = d - COP.z
// dx = x2 - x1 = 2
// dy = y2 - y1 = dx / aspect
// px = x2 + x1 = 0
// py = y2 + y1 = 0
// Thus we have
float dx, dy, D;
D = d;
dx = 0.63;
dy = dx/aspect;

// Transposed
regularPyramid[0] = vec4(2.0*D/dx, 0, 0, 0);
regularPyramid[1] = vec4(0, 2.0*D/dy, 0, 0);
regularPyramid[2] = vec4(0, 0, 1.0, 0);
regularPyramid[3] = vec4(0, 0, 0, 1.0);

// M(Scale) =
// 1/D  0   0   0
//  0  1/D  0   0
//  0   0  1/D  0
//  0   0   0   1

scale[0] = vec4(1.0/D, 0, 0, 0);
scale[1] = vec4(0, 1.0/D, 0, 0);
scale[2] = vec4(0, 0, 1.0/D, 0);
scale[3] = vec4(0, 0, 0, 1.0);

// M(Canonical) =
//  1   0   0   0
//  0   1   0   0
//  0   0   1   1
//  0   0   0   1

canonical[0] = vec4(1.0, 0, 0, 0);
canonical[1] = vec4(0, 1.0, 0, 0);
canonical[2] = vec4(0, 0, 1.0, 0);
canonical[3] = vec4(0, 0, 1.0, 1.0);

projectionMatrix = translationUV * regularPyramid * scale * canonical;

return projectionMatrix;
#endif
return projectionMatrix;
}

// Used to generate a simple "look-at" camera.
mat4 computeViewMatrix(vec3 VRP, vec3 TP, vec3 VUV) {
mat4 viewMatrix = mat4(1);

#ifdef PROJECTION
// Put your code here
// Calculate the VPN by getting the vector from the view reference plane to the target point
vec3 VPN = normalize(TP - VRP);
vec3 n = VPN;
vec3 u = normalize(cross(VUV, n));
vec3 v = normalize(cross(n, u));
vec3 t = vec3(-dot(VRP,u),-dot(VRP,v),-dot(VRP,n));

// M (View)	=
//  u.x  v.x  n.x  0
//  u.y  v.y  n.y  0
//  u.z  v.z  n.z  0
//  t.x  t.y  t.z  1

// Populate the view matrix with values
viewMatrix[0]  = vec4(u.x, u.y, u.z, t.x);
viewMatrix[1]  = vec4(v.x, v.y, v.z, t.y);
viewMatrix[2]  = vec4(n.x, n.y, n.z, t.z);
viewMatrix[3]  = vec4(0, 0, 0, 1.0);

#endif
return viewMatrix;
}

vec3 getCameraPosition() {
#ifdef ANIMATION
// Put your code here
return vec3(8.0*sin(time), 8.0*cos(time), 8.0*cos(time));
#else
return vec3(0, 0, 10);
#endif
}

// Takes a single input vertex and projects it using the input view and projection matrices
vec3 projectVertexPosition(vec3 position) {

// Set the parameters for the look-at camera.
vec3 TP = vec3(0, 0, 0);
vec3 VRP = getCameraPosition();
vec3 VUV = vec3(0, 1, 0);

// Compute the view matrix.
mat4 viewMatrix = computeViewMatrix(VRP, TP, VUV);

// Compute the projection matrix.
mat4 projectionMatrix = computeProjectionMatrix();

#ifdef PROJECTION
// Put your code here
// Transfrom from the WC to VC
vec4 vec4pos = vec4(position, 1.0) * viewMatrix;

// Transform from VC to canonical projection space
vec4 newVec4pos = vec4pos * projectionMatrix;

// Homogeneous point to Euclidean 3D point
// Since this is a point relative to projection space, the projection of this point to X-Y plane
// is clearly obtained by ignoring the z component (textbook Computer Graphics And Virtual Environments: From Realism to Real-Time, page 201)
// And since the z has already been projected to the projection plane, its value should remain altered.
float z = newVec4pos.z;
position = vec3(newVec4pos)/newVec4pos.w;
position.z = z;

return position;
#else
return position;
#endif
}

// Projects all the vertices of a polygon
void projectPolygon(inout Polygon projectedPolygon, Polygon polygon) {
copyPolygon(projectedPolygon, polygon);
for (int i = 0; i < MAX_VERTEX_COUNT; ++i) {
if (i < polygon.vertexCount) {
projectedPolygon.vertices[i].position = projectVertexPosition(polygon.vertices[i].position);
}
}
}

// Draws a polygon by projecting, clipping, ratserizing and interpolating it
void drawPolygon(
vec2 point,
Polygon clipWindow,
Polygon oldPolygon,
inout vec3 color,
inout float depth)
{
Polygon projectedPolygon;
projectPolygon(projectedPolygon, oldPolygon);

Polygon clippedPolygon;
sutherlandHodgmanClip(projectedPolygon, clipWindow, clippedPolygon);

if (isPointInPolygon(point, clippedPolygon)) {

Vertex interpolatedVertex =
interpolateVertex(point, projectedPolygon);
#if defined(ZBUFFERING)
// Put your code here
// According to lecture slide:
// If current z < zbuffer then set color and zbuffer
// Or else, do nothing
if (interpolatedVertex.position.z < depth){
color = interpolatedVertex.color;
depth = interpolatedVertex.position.z;
}
#else
// Put your code to handle z buffering here
color = interpolatedVertex.color;
depth = interpolatedVertex.position.z;
#endif
}

if (isPointOnPolygonVertex(point, clippedPolygon)) {
color = vec3(1);
}
}

// Main function calls

void drawScene(vec2 pixelCoord, inout vec3 color) {
color = vec3(0.3, 0.3, 0.3);

// Convert from GL pixel coordinates 0..N-1 to our screen coordinates -1..1
vec2 point = 2.0 * pixelCoord / vec2(viewport) - vec2(1.0);

Polygon clipWindow;
clipWindow.vertices[0].position = vec3(-0.65,  0.95, 1.0);
clipWindow.vertices[1].position = vec3( 0.65,  0.75, 1.0);
clipWindow.vertices[2].position = vec3( 0.75, -0.65, 1.0);
clipWindow.vertices[3].position = vec3(-0.75, -0.85, 1.0);
clipWindow.vertexCount = 4;

// Draw the area outside the clip region to be dark
color = isPointInPolygon(point, clipWindow) ? vec3(0.5) : color;

const int triangleCount = 2;
Polygon triangles[triangleCount];

triangles[0].vertices[0].position = vec3(-2, -2, 0.0);
triangles[0].vertices[1].position = vec3(4, 0, 3.0);
triangles[0].vertices[2].position = vec3(-1, 2, 0.0);
triangles[0].vertices[0].color = vec3(1.0, 0.5, 0.2);
triangles[0].vertices[1].color = vec3(0.8, 0.8, 0.8);
triangles[0].vertices[2].color = vec3(0.2, 0.5, 1.0);
triangles[0].vertexCount = 3;

triangles[1].vertices[0].position = vec3(3.0, 2.0, -2.0);
triangles[1].vertices[2].position = vec3(0.0, -2.0, 3.0);
triangles[1].vertices[1].position = vec3(-1.0, 2.0, 4.0);
triangles[1].vertices[1].color = vec3(0.2, 1.0, 0.1);
triangles[1].vertices[2].color = vec3(1.0, 1.0, 1.0);
triangles[1].vertices[0].color = vec3(0.1, 0.2, 1.0);
triangles[1].vertexCount = 3;

float depth = 10000.0;
// Project and draw all the triangles
for (int i = 0; i < triangleCount; i++) {
drawPolygon(point, clipWindow, triangles[i], color, depth);
}
}

void main() {
drawScene(gl_FragCoord.xy, gl_FragColor.rgb);
gl_FragColor.a = 1.0;
}
