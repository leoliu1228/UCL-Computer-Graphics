#define SOLUTION_CYLINDER_AND_PLANE
#define SOLUTION_MATERIAL
#define SOLUTION_SHADOW
#define SOLUTION_REFLECTION_REFRACTION
#define SOLUTION_FRESNEL

precision highp float;
uniform float time;

struct PointLight {
    vec3 position;
    vec3 color;
};

struct Material {
    vec3  diffuse;
    vec3  specular;
    float glossiness;
#ifdef SOLUTION_MATERIAL
	// Put the variables for reflection and refraction here
  float reflectionWeight;
  float refractionWeight;
  float refractionIndex;
#endif
};

struct Sphere {
    vec3 position;
    float radius;
    Material material;
};

struct Plane {
    vec3 normal;
    float d;
    Material material;
};

struct Cylinder {
    vec3 position;
    vec3 direction;
    float radius;
    Material material;
};

const int lightCount = 2;
const int sphereCount = 3;
const int planeCount = 1;
const int cylinderCount = 2;

struct Scene {
    vec3 ambient;
    PointLight[lightCount] lights;
    Sphere[sphereCount] spheres;
    Plane[planeCount] planes;
    Cylinder[cylinderCount] cylinders;
};

struct Ray {
    vec3 origin;
    vec3 direction;
};

// Contains all information pertaining to a ray/object intersection
struct HitInfo {
    bool hit;
    float t;
    vec3 position;
    vec3 normal;
    Material material;
};

HitInfo getEmptyHit() {
	return HitInfo(
      	false,
      	0.0,
      	vec3(0.0),
      	vec3(0.0),
#ifdef SOLUTION_MATERIAL
		// Update the constructor call
	   Material(vec3(0.0), vec3(0.0), 1.0, 1.0, 0.0, 1.0)
#else
		Material(vec3(0.0), vec3(0.0), 0.0)
#endif
	);
}

// Sorts the two t values such that t1 is smaller than t2
void sortT(inout float t1, inout float t2) {
  	// Make t1 the smaller t
    if (t2 < t1) {
		float temp = t1;
		t1 = t2;
		t2 = temp;
    }
}

// Tests if t is in an interval
bool isTInInterval(const float t, const float tMin, const float tMax) {
	return t > tMin && t < tMax;
}

// Get the smallest t in an interval
bool getSmallestTInInterval(float t0, float t1, const float tMin, const float tMax, inout float smallestTInInterval) {
	sortT(t0, t1);
	// As t0 is smaller, test this first
	if (isTInInterval(t0, tMin, tMax)) {
		smallestTInInterval = t0;
        return true;
	}

	// If t0 was not in the interval, still t1 could be
	if (isTInInterval(t1, tMin, tMax)) {
		smallestTInInterval = t1;
		return true;
	}
	// None was
	return false;
}

HitInfo intersectSphere(const Ray ray, const Sphere sphere, const float tMin, const float tMax) {

    vec3 to_sphere = ray.origin - sphere.position;

    float a = dot(ray.direction, ray.direction);
	  float b = 2.0 * dot(ray.direction, to_sphere);
    float c = dot(to_sphere, to_sphere) - sphere.radius * sphere.radius;
    float D = b * b - 4.0 * a * c;
    if (D > 0.0)
    {
		float t0 = (-b - sqrt(D)) / (2.0 * a);
		float t1 = (-b + sqrt(D)) / (2.0 * a);

      	float smallestTInInterval;
      	if (!getSmallestTInInterval(t0, t1, tMin, tMax, smallestTInInterval)) {
			return getEmptyHit();
        }

      	vec3 hitPosition = ray.origin + smallestTInInterval * ray.direction;

      	vec3 normal =
			length(ray.origin - sphere.position) < sphere.radius + 0.001 ?
          	-normalize(hitPosition - sphere.position) :
      		normalize(hitPosition - sphere.position);

        return HitInfo(
          	true,
          	smallestTInInterval,
          	hitPosition,
          	normal,
          	sphere.material
        );
    }
    return getEmptyHit();
}

HitInfo intersectPlane(const Ray ray,const Plane plane, const float tMin, const float tMax) {
	#ifdef SOLUTION_CYLINDER_AND_PLANE
	// Add your plane intersection code here

	// To check if the hit position is on the plane, we need to calculate whether it is perpendicular to the normal of the plane
	// In this case their dot product must be 0: Vector(PP')⋅ Vector(N) = 0
	// PP' = O + tD - P'(Plane.position which will be calculated later) = O - P' + tD
	// N = N(Plane.nromal)
	// Thus we have Vector(PP')⋅ Vector(N) = Vector(O - P' + tD)⋅ Vector(Normal) = [Vector(O - P') + t*Vector(D)]⋅ Vector(Normal) = Vector(O - P')⋅ Vector(Normal) + t*Vector(D)⋅ Vector(Normal) = 0
  // Now we have Vector(O - P')⋅ Vector(Normal) + t*Vector(D)⋅ Vector(Normal) = 0
  // Vector(O - P')⋅ Vector(Normal) = -t*Vector(D)⋅ Vector(Normal)
  // And we can rewrite this function as t = - Vector(O - P')⋅ Vector(Normal) / Vector(D)⋅ Vector(Normal) = Vector(P' - O)⋅ Vector(Normal) / Vector(D)⋅ Vector(Normal)
  // Here we abbreviate it as t = a/b

	vec3 planePosition = plane.normal * -plane.d; // Although d is a positive number, the actual displacement is following the opposite direction. e.g 4.5 gives (0, -4.5, 0), thus we need let it have the same orientation as how normal is calculated
	float a = dot(planePosition,plane.normal) - dot(ray.origin,plane.normal);
	float b = dot(ray.direction, plane.normal);

  // Make sure the equation is defined
  if (b != 0.0){
    float t = a/b;

    // There is only one solution so we just need to check if the t is within the tMin tMax range
    if (!isTInInterval(t, tMin, tMax)) {
  			return getEmptyHit();
          }

  	// According to P = O + tD
		vec3 hitPosition = ray.origin + t * ray.direction ;
    // We need to check if the ray is below the plane.
    // If so we need to invert its normal
    // Since the plane only moves along y-axis in this case, we simply compare ray's y value to the plane's y value.
    vec3 normal = ray.origin.y > planePosition.y?
    plane.normal:-plane.normal;

  	return HitInfo(
            	true,
            	t,
            	hitPosition,
            	normal,
            	plane.material
  		);
  }

  // When b is not defined, return empty hit
	return getEmptyHit();

	#endif
    return getEmptyHit();
}

float lengthSquared(vec3 x) {
	return dot(x, x);
}

HitInfo intersectCylinder(const Ray ray, const Cylinder cylinder, const float tMin, const float tMax) {
	#ifdef SOLUTION_CYLINDER_AND_PLANE
	// Raytracing a cylinder is similar to sphere except that it consists of infinite layers of circles on one given direction (based on the normal)
  // Any point to the central axis of cylinder that its distance is equal to the radius (which also means it is on the circle) is considered as the hit position on the surface
	// Thus any point on the circle within the cylinder can be found using ((P - C(Cylinder.position)) × D'(Cylinder.direction))^2 = Radius^2
  // For a better demonstration (may take some time to load the image): http://yuqi.ninja/ucl/cg/cw1/cylinder.png
	// Then we put known variables into the equation
  // ((O + tD - C) × D')^2 =  ((O - C + tD) × D')^2 = ((to_Cylinder + tD) × D')^2 = r^2
  // Next we have (to_Cylinder × D' + t * D × D')^2 - r^2 = 0, and it can be abbreviated as (h + t * i )^2 - j = 0
  // Finally we have i^2 * t^2 + 2thi + h^2 - j = 0
	// And it is just another at^2 + 2bt + c = 0

  vec3 to_cylinder = ray.origin - cylinder.position;
  vec3 h = cross(to_cylinder,cylinder.direction);
  vec3 i = cross(ray.direction,cylinder.direction);
  float j = cylinder.radius * cylinder.radius;

  float a = dot(i,i);
  float b = 2.0 * dot(h,i);
  float c = dot(h,h) - j;

  float D = b * b - 4.0 * a * c;

	if (D > 0.0)
    {
		float t0 = (-b - sqrt(D)) / (2.0 * a);
		float t1 = (-b + sqrt(D)) / (2.0 * a);

    float smallestTInInterval;
    if (!getSmallestTInInterval(t0, t1, tMin, tMax, smallestTInInterval)) {
      return getEmptyHit();
    }

    vec3 hitPosition = ray.origin + smallestTInInterval * ray.direction;

    // Since we know the hit position, it is easier to calculate the normal with vectors
    // We know the hit position P, O(Ray.origin) and C(Cylinder.position), D(Cylinder.direction), and assume that the P is on the one layer of circle with center S (and S is also on the D direction), and we will eliminate S in later stage
    // Then the normal will be Vector(SP), which is perpendicular to D
    // Therefore we have vector(SP) = Vector(CP) - Vector(CS) = Vector(OP) - Vector(OC) - Vector(CS)
    // We can easily calculate Vector(OP) and Vector(OC) since we know all involved values
    // Then Vector(CS) = (Vector(SP)⋅ D) * D which only contains known variables
    // Thus we have the Normal = Vector(OP) - Vector(OC) - (Vector(SP) ⋅ D) * D
    // For a better demonstration (may take some time to load the image): http://yuqi.ninja/ucl/cg/cw1/cylinder.png

    vec3 OP = hitPosition - ray.origin;
    vec3 OC = cylinder.position - ray.origin;
    vec3 CS = dot((OP-OC),cylinder.direction) * cylinder.direction;
    vec3 normal = normalize(OP-OC-CS);

    // Before output the normal we also need to know if the ray is inside the cylinder
    // If it is inside the cylinder, we invert its normal otherwise proceed as usual
    // The checking process is exactly same as normal calculation except this time we check the distance between ray origin and central axis of the cylinder
    // We simply replace the P as O, in this case Vector(OP) = 0.0
    // Vector(OS) = Vector(OC) + Vector(CS)
    vec3 OS = OC + dot(0.0-OC,cylinder.direction) * cylinder.direction;
    normal = dot(OS,OS) < cylinder.radius + 0.001?
    -normal : normal;

    return HitInfo(
        true,
        smallestTInInterval,
        hitPosition,
        normal,
        cylinder.material
        );
    }
	#else
		return getEmptyHit();
	#endif
    	return getEmptyHit();
}

HitInfo getBetterHitInfo(const HitInfo oldHitInfo, const HitInfo newHitInfo) {
	if(newHitInfo.hit)
  		if(newHitInfo.t < oldHitInfo.t)  // No need to test for the interval, this has to be done per-primitive
          return newHitInfo;
  	return oldHitInfo;
}

HitInfo intersectScene(const Scene scene, const Ray ray, const float tMin, const float tMax) {
	HitInfo bestHitInfo;
	bestHitInfo.t = tMax;
	bestHitInfo.hit = false;
	for (int i = 0; i < cylinderCount; ++i) {
    	bestHitInfo = getBetterHitInfo(bestHitInfo, intersectCylinder(ray, scene.cylinders[i], tMin, tMax));
	}
	for (int i = 0; i < sphereCount; ++i) {
		bestHitInfo = getBetterHitInfo(bestHitInfo, intersectSphere(ray, scene.spheres[i], tMin, tMax));
	}
	for (int i = 0; i < planeCount; ++i) {
		bestHitInfo = getBetterHitInfo(bestHitInfo, intersectPlane(ray, scene.planes[i], tMin, tMax));
	}
	return bestHitInfo;
}

vec3 shadeFromLight(
	const Scene scene,
	const Ray ray,
	const HitInfo hit_info,
	const PointLight light)
{
	vec3 hitToLight = light.position - hit_info.position;

	vec3 lightDirection = normalize(hitToLight);
	vec3 viewDirection = normalize(hit_info.position - ray.origin);
	vec3 reflectedDirection = reflect(viewDirection, hit_info.normal);
	float diffuse_term = max(0.0, dot(lightDirection, hit_info.normal));
	float specular_term  = pow(max(0.0, dot(lightDirection, reflectedDirection)), hit_info.material.glossiness);

	#ifdef SOLUTION_SHADOW

  // To check if the light is visible from the point being shaded, we just need to know if the light to hit point is blocked by any object in the scene.
  // So we construct a new shadow ray and bring it back to the intersection function.
  // If shadow the ray does hit something in the scene, that means it can not be seen.
  // Thus the visibility will be ZERO which forms the shadow.

  Ray shadowRay;
  shadowRay.origin = hit_info.position;
  shadowRay.direction = hitToLight;
  HitInfo shadowHitInfo = intersectScene(scene, shadowRay, 0.001, 100000.0);
  float visibility = shadowHitInfo.hit && shadowHitInfo.t > 0.0 && shadowHitInfo.t < 1.0 ?
    0.0 : 1.0;

  // Typical errors that may occur here:
  // Only checking whether the light collides with any object in the scene is not enough, we also have to limit the t value because:
  // In a regular ray tracing scenario, we know the origin and the direction to the object, but we do not know how far is the object away from the ray, that is why we need to know the t values
  // But for the shadow ray, we know the light position and hit position
  // Thus the distance is known and we only want to know within this distance (scale from 0 to 1, 1 = full distance 0.5 is like half way) if there is an object blocking the light

	#else
  float visibility = 1.0;
	#endif

	Ray mirrorRay;
	mirrorRay.origin = hit_info.position;
	mirrorRay.direction = reflect(lightDirection, hit_info.normal);
	HitInfo mirrorHitInfo = intersectScene(scene, mirrorRay, 0.001, 100000.0);

  return visibility *
		 light.color * (
		 specular_term * hit_info.material.specular +
		 diffuse_term * hit_info.material.diffuse);
}

vec3 background(const Ray ray) {
	// A simple implicit sky that can be used for the background
	return vec3(0.2) + vec3(0.8, 0.6, 0.5) * max(0.0, ray.direction.y);
}

// It seems to be a WebGL issue that the third parameter needs to be inout instea dof const on Tobias' machine
vec3 shade(const Scene scene, const Ray ray, inout HitInfo hitInfo) {

  	if(!hitInfo.hit) {
		return background(ray);
  	}

    vec3 shading = scene.ambient * hitInfo.material.diffuse;
    for (int i = 0; i < lightCount; ++i) {
		shading += shadeFromLight(scene, ray, hitInfo, scene.lights[i]);
    }
    return shading;
}

Ray getFragCoordRay(const vec2 frag_coord) {
	float sensorDistance = 1.0;
  	vec2 sensorMin = vec2(-1, -0.5);
  	vec2 sensorMax = vec2(1, 0.5);
  	vec2 pixelSize = (sensorMax- sensorMin) / vec2(800, 400);
  	vec3 origin = vec3(0, 0, sensorDistance);
    vec3 direction = normalize(vec3(sensorMin + pixelSize * frag_coord, -sensorDistance));

  	return Ray(origin, direction);
}

float fresnel(const vec3 viewDirection, const vec3 normal) {
#ifdef SOLUTION_FRESNEL
	// Put your code to compute the Fresnel effect here
  // According to Schlick's approximation, the Fresnel factor = R0 + (1 - R0)(1 - cos(theta))^5 where R0 is the Reflection Coefficient
  // R0 = (eta1 - eta2) / (eta1 + eta2) where eta is the IOR
  // However, in this function we only take view direction and normal as input parameters
  // Thus we need to estimate the reflection coefficient and the power based on objects in the scene (And it is also the reason why we have to adjust IOR of glass in this case)
  // We simply treat every object as vacuum and after some testing we decrease the power from 5.0 to aroudn 1.8 to match the provided visual effects
  float R0 = 0.0;
	return R0 + (1.0 - R0) * pow(1.0 - dot(viewDirection, normal), 1.8);
#else
	return 1.0;
#endif
}

vec3 colorForFragment(const Scene scene, const vec2 fragCoord) {

    Ray initialRay = getFragCoordRay(fragCoord);
  	HitInfo initialHitInfo = intersectScene(scene, initialRay, 0.001, 10000.0);
  	vec3 result = shade(scene, initialRay, initialHitInfo);

  	Ray currentRay;
  	HitInfo currentHitInfo;

  	// Compute the reflection
  	currentRay = initialRay;
  	currentHitInfo = initialHitInfo;

  	// The initial strength of the reflection
  	float reflectionWeight = 1.0;

  	const int maxReflectionStepCount = 2;
  	for (int i = 0; i < maxReflectionStepCount; i++) {
		if (!currentHitInfo.hit) break;

#ifdef SOLUTION_REFLECTION_REFRACTION
		// Put your reflection weighting code here
    reflectionWeight *= currentHitInfo.material.reflectionWeight;
#endif

#ifdef SOLUTION_FRESNEL
		// Add Fresnel contribution
    reflectionWeight *= fresnel(-currentRay.direction,currentHitInfo.normal);

#else
		reflectionWeight *= 0.5;
#endif

		Ray nextRay;
#ifdef SOLUTION_REFLECTION_REFRACTION
		// Put your code to compute the reflection ray here
    // According to the equation on lecture slides: r = -e + 2 * (normal ⋅ e) * normal. (e is opposite to currentRay's direction here so we need to add a negative sign to change its direction)
    nextRay.origin = currentHitInfo.position;
    nextRay.direction = currentRay.direction + 2.0 * dot(currentHitInfo.normal, -currentRay.direction) * currentHitInfo.normal;

#endif
		currentRay = nextRay;
		currentHitInfo = intersectScene(scene, currentRay, 0.001, 10000.0);
		result += reflectionWeight * shade(scene, currentRay, currentHitInfo);
    }

	// Compute the refraction
	currentRay = initialRay;
	currentHitInfo = initialHitInfo;

  	// The initial medium is air
  	float currentIOR = 1.0;

  	// The initial strength of the refraction.
  	float refractionWeight = 1.0;

  	const int maxRefractionStepCount = 2;
  	for(int i = 0; i < maxRefractionStepCount; i++) {

#ifdef SOLUTION_REFLECTION_REFRACTION
		// Put your refraction weighting code here
		refractionWeight *= currentHitInfo.material.refractionWeight;
#else
		refractionWeight *= 0.5;
#endif

#ifdef SOLUTION_FRESNEL
		// Add Fresnel contribution
    // According to the assumption F(Refraction) = 1 - F(Reflection)
    refractionWeight *= (1.0 - fresnel(-currentRay.direction,currentHitInfo.normal));
#endif

		Ray nextRay;
#ifdef SOLUTION_REFLECTION_REFRACTION
		// Put your code to compute the reflection ray and track the IOR

    nextRay.origin = currentHitInfo.position;
    // According to Snell's Law
    float alpha = dot(-currentRay.direction,currentHitInfo.normal);
    float materialIOR = currentHitInfo.material.refractionIndex;

    // Because the next hit position may still be the same material (e.g. inside surface of an object), which means there is no other object colliding with the current object.
    // Therefore, we need to make sure the currentIOR is changed back to the air/vacuum.
    currentIOR = materialIOR == currentIOR? 1.0:currentIOR;

    float eta = currentIOR/materialIOR;

    // Then we record the current IOR
    currentIOR = materialIOR;

    float refractionRoot = 1.0 + eta * eta * (alpha * alpha - 1.0);

    if (refractionRoot < 0.0){
      // If the root is negative then the Total Internal Reflection occurs
      // Break because we don't need to calculate it
      break;

    }else{
      // Refraction occurs here
      nextRay.origin = currentHitInfo.position;
      nextRay.direction =  eta * currentRay.direction + currentHitInfo.normal * (eta * alpha - sqrt(refractionRoot));
      currentRay = nextRay;
    }

#endif

		currentHitInfo = intersectScene(scene, currentRay, 0.001, 10000.0);
		result += refractionWeight * shade(scene, currentRay, currentHitInfo);

		if (!currentHitInfo.hit) break;
	}
	return result;
}

Material getDefaultMaterial() {
	#ifdef SOLUTION_MATERIAL
	// Update the default material call to match the new parameters of Material
  // Default material.
  Material defaultMaterial;
	defaultMaterial.diffuse = vec3(0.3);
	defaultMaterial.specular = vec3(0);
	defaultMaterial.glossiness = 1.0;
  defaultMaterial.reflectionWeight = 1.0; // Default value since no specific property provided.
  defaultMaterial.refractionWeight = 0.0; // Opaque material
  defaultMaterial.refractionIndex = 1.0; // Opaque material
	return defaultMaterial;

	#else
	return Material(vec3(0.3), vec3(0), 1.0);
	#endif
}

Material getPaperMaterial() {
	#ifdef SOLUTION_MATERIAL
	// Replace by your definition of a paper material
	Material paper;
	paper.diffuse = vec3(0.67, 0.67, 0.67); // RGB(160,160,160). A little bit gray makes the paper look even more realistic rather than using pure white. RGB(150,150,150)
	paper.specular = vec3(0.0,0.0,0.0); // Shiness and highlight color are not really applicable for paper material
	paper.glossiness = 1.0; // Glossiness defines how clear the reflections are while paper does not reflect
  paper.reflectionWeight = 0.0; // Regular paper will not reflect
  paper.refractionWeight = 0.0; // Opaque material
  paper.refractionIndex = 1.0; // Opaque material
	return paper;
	#else
    return getDefaultMaterial();
	#endif
}

Material getPlasticMaterial() {
	#ifdef SOLUTION_MATERIAL
	// Replace by your definition of a plastic material
	Material plastic;
	plastic.diffuse =vec3(0.90, 0.31, 0.09); // Yellow~Orange RGB(230,80,24).
	plastic.specular = vec3(0.75,0.75,0.75); // Well polished plastic is usually shinny but will not be as shinny as metal materials such as gold
	plastic.glossiness = 8.0; // The lighting effects on its surface should be blurry thus we need increase the glossiness a little bit
  plastic.reflectionWeight = 0.9; // Plastic surface does reflect. 0.9 looks good enough for reaching the visual effects.
  plastic.refractionWeight = 0.0; // Opaque material
  plastic.refractionIndex = 1.0; // Opaque material
	return plastic;
	#else
    return getDefaultMaterial();
	#endif
}

Material getGlassMaterial() {
	#ifdef SOLUTION_MATERIAL
	// Replace by your definition of a glass material
	Material glass;
	glass.diffuse =vec3(0.0, 0.0, 0.0); // Transparent material. No color required.
	glass.specular = vec3(1.0,1.0,1.0); // Glass surface is so smooth that it only has very little spreading of reflected highlights and blurrings
	glass.glossiness = 100000.0; // In reality the glass should refelct a very sharp portion of highlights (like a glowing spot).
                             // Values like around 2000 will show such effects but here we use 100000 to match the final effect
  glass.reflectionWeight = 1.0; // Glass material is both refractive and reflective.
  glass.refractionWeight = 1.0; // Transparent material (Translucent if it is frosted glass)
  glass.refractionIndex = 1.28; // In physics, the common IOR of glass is 1.52. However the calculation in this scene is not as accurate as physically based rendering since
                                // a lot of approximations have been used in this ray tracer. And most importantly, a properly done glass material should refract the object behind it
                                // upside down. Values like 4.0 will reach such effects.
                                // In order to match the final effect as shown in coursework requirement, 1.28 is chosen for the best visial effect.
	return glass;
	#else
    return getDefaultMaterial();
	#endif
}

Material getSteelMirrorMaterial() {
	#ifdef SOLUTION_MATERIAL
	// Replace by your definition of a steel mirror material
	Material steelMirror;
	steelMirror.diffuse =vec3(0.12, 0.12, 0.12); // Dark base color RGB(30,30,30)
	steelMirror.specular = vec3(0.0,0.0,0.0); // Mirror sounds like a shinny material but in fact it basically only reflect the scene and light.
                                            // The mirror itself will not be glowing
	steelMirror.glossiness = 1.0; // Normally mirror surface is extremtly smooth so no burrings
  steelMirror.reflectionWeight = 0.8; // Mirror does reflect but 1.0 is a bit high since this scene has a lot of refelction and refraction going on which makes its surface too bright
                                      // Decrease to 0.8 to match the final visual effects
  steelMirror.refractionWeight = 0.0; // Opaque material
  steelMirror.refractionIndex = 1.0; // Opaque material
	return steelMirror;
	#else
    return getDefaultMaterial();
	#endif
}

vec3 tonemap(const vec3 radiance) {
	const float monitorGamma = 2.0;
	return pow(radiance, vec3(1.0 / monitorGamma));
}

void main()
{
    // Setup scene
	Scene scene;
  	scene.ambient = vec3(0.12, 0.15, 0.2);

    // Lights
    scene.lights[0].position          = vec3(5, 15, -5);
    scene.lights[0].color             = 0.5 * vec3(0.9, 0.5, 0.1);

  	scene.lights[1].position          = vec3(-15, 5, 2);
    scene.lights[1].color             = 0.5 * vec3(0.1, 0.3, 1.0);

    // Primitives
    scene.spheres[0].position         = vec3(10, -5, -16);
    scene.spheres[0].radius           = 6.0;
    scene.spheres[0].material 				= getPaperMaterial();

  	scene.spheres[1].position         = vec3(-7, -1, -13);
    scene.spheres[1].radius          	= 4.0;
    scene.spheres[1].material				  = getPlasticMaterial();

    scene.spheres[2].position        	= vec3(0, 0.5, -5);
    scene.spheres[2].radius           = 2.0;
    scene.spheres[2].material   			= getGlassMaterial();

  	scene.planes[0].normal            = vec3(0, 1, 0);
  	scene.planes[0].d              	  = 4.5;
    scene.planes[0].material				  = getSteelMirrorMaterial();

  	scene.cylinders[0].position       = vec3(-1, 1, -18);
  	scene.cylinders[0].direction      = normalize(vec3(-1, 2, -1));
  	scene.cylinders[0].radius         = 1.5;
    scene.cylinders[0].material				= getPaperMaterial();

  	scene.cylinders[1].position       = vec3(4, 1, -5);
  	scene.cylinders[1].direction      = normalize(vec3(1, 4, 1));
  	scene.cylinders[1].radius         = 0.4;
    scene.cylinders[1].material				= getPlasticMaterial();

	// compute color for fragment
	gl_FragColor.rgb = tonemap(colorForFragment(scene, gl_FragCoord.xy));
	gl_FragColor.a = 1.0;
}
