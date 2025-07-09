#include <glm/glm.hpp>
#include <glm/gtc/random.hpp>
#include <glm/gtc/constants.hpp>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#include <vector>
#include <memory>
#include <random>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <omp.h>

using namespace glm;

struct Ray {
    vec3 origin;
    vec3 direction;
};

struct Material {
    vec3 albedo = vec3(0.8f);
    float metallic = 0.0f;
    float roughness = 0.5f;
    float ior = 1.5f;
    float transmission = 0.0f;
    float clearcoat = 0.0f;
    float clearcoatRoughness = 0.03f;
    float subsurfaceWeight = 0.0f;
    vec3 emissive = vec3(0.0f);

    float getComplexity() const {
        float complexity = 0.1f;
        if (transmission > 0.0f) complexity += 0.4f * transmission;
        if (metallic > 0.0f) complexity += 0.2f * metallic;
        if (roughness < 0.3f) complexity += 0.3f * (1.0f - roughness / 0.3f);
        if (clearcoat > 0.0f) complexity += 0.3f * clearcoat;
        if (subsurfaceWeight > 0.0f) complexity += 0.5f * subsurfaceWeight;
        if (ior != 1.5f) complexity += 0.2f * abs(ior - 1.5f);
        return clamp(complexity, 0.1f, 1.0f);
    }

    bool isLight() const {
        return length(emissive) > 0.0f;
    }
};

struct HitInfo {
    bool hit = false;
    float t = std::numeric_limits<float>::max();
    vec3 position;
    vec3 normal;
    Material material;
};

class Shape {
public:
    virtual HitInfo intersect(const Ray& ray) const = 0;
    virtual vec3 sampleSurface(float u1, float u2, vec3& normal, float& pdf) const = 0;
    virtual float area() const = 0;
    virtual ~Shape() = default;
};

class Sphere : public Shape {
    vec3 center;
    float radius;
    Material material;

public:
    Sphere(const vec3& c, float r, const Material& m)
        : center(c), radius(r), material(m) {}

    HitInfo intersect(const Ray& ray) const override {
        HitInfo info;
        vec3 oc = ray.origin - center;
        float a = dot(ray.direction, ray.direction);
        float b = 2.0f * dot(oc, ray.direction);
        float c = dot(oc, oc) - radius * radius;
        float discriminant = b * b - 4.0f * a * c;

        if (discriminant < 0) return info;

        float t = (-b - sqrt(discriminant)) / (2.0f * a);
        if (t < 0.001f) {
            t = (-b + sqrt(discriminant)) / (2.0f * a);
            if (t < 0.001f) return info;
        }

        info.hit = true;
        info.t = t;
        info.position = ray.origin + t * ray.direction;
        info.normal = normalize(info.position - center);
        info.material = material;
        return info;
    }

    vec3 sampleSurface(float u1, float u2, vec3& normal, float& pdf) const override {
        float z = 1.0f - 2.0f * u1;
        float r = sqrt(max(0.0f, 1.0f - z * z));
        float phi = 2.0f * pi<float>() * u2;

        normal = vec3(r * cos(phi), r * sin(phi), z);
        vec3 point = center + radius * normal;
        pdf = 1.0f / area();

        return point;
    }

    float area() const override {
        return 4.0f * pi<float>() * radius * radius;
    }

    const Material& getMaterial() const { return material; }
};

struct LightInfo {
    const Shape* shape;
    vec3 emission;
    float area;
};

class Scene {
    std::vector<std::unique_ptr<Shape>> shapes;
    std::vector<LightInfo> lights;

public:
    void add(std::unique_ptr<Shape> shape) {
        if (auto* sphere = dynamic_cast<Sphere*>(shape.get())) {
            if (sphere->getMaterial().isLight()) {
                LightInfo light;
                light.shape = shape.get();
                light.emission = sphere->getMaterial().emissive;
                light.area = shape->area();
                lights.push_back(light);
            }
        }
        shapes.push_back(std::move(shape));
    }

    HitInfo intersect(const Ray& ray) const {
        HitInfo closest;
        for (const auto& shape : shapes) {
            HitInfo info = shape->intersect(ray);
            if (info.hit && info.t < closest.t) {
                closest = info;
            }
        }
        return closest;
    }

    bool visible(const vec3& p1, const vec3& p2) const {
        vec3 dir = p2 - p1;
        float dist = length(dir);
        dir /= dist;

        Ray shadowRay{p1 + dir * 0.001f, dir};
        HitInfo hit = intersect(shadowRay);

        return !hit.hit || hit.t > dist - 0.002f;
    }

    const std::vector<LightInfo>& getLights() const { return lights; }

    vec3 sampleLight(float rand, vec3& position, vec3& normal, float& pdf) const {
        if (lights.empty()) {
            pdf = 0.0f;
            return vec3(0.0f);
        }

        float totalArea = 0.0f;
        for (const auto& light : lights) {
            totalArea += light.area;
        }

        float target = rand * totalArea;
        float accumulated = 0.0f;

        for (const auto& light : lights) {
            accumulated += light.area;
            if (accumulated >= target) {
                float u1 = fract(rand * 1000.0f);
                float u2 = fract(rand * 10000.0f);

                float shapePdf;
                position = light.shape->sampleSurface(u1, u2, normal, shapePdf);
                pdf = shapePdf * (light.area / totalArea);

                return light.emission;
            }
        }

        pdf = 0.0f;
        return vec3(0.0f);
    }
};

struct Sampler {
    int pixelIndex;
    int sampleIndex;
    int dimension;

    Sampler(int pixel, int sample) : pixelIndex(pixel), sampleIndex(sample), dimension(0) {}

    float get1D() {
        return stratifiedSample1D(pixelIndex, sampleIndex, dimension++);
    }

    vec2 get2D() {
        float u = get1D();
        float v = get1D();
        return vec2(u, v);
    }

private:
    float stratifiedSample1D(int pixel, int sample, int dim) const {
        const int strataCount = 16;
        int stratum = (pixel * 7 + sample * 13 + dim * 29) % strataCount;

        uint32_t hash = pixel * 73856093u ^ sample * 19349663u ^ dim * 83492791u;
        hash = hash * 1103515245u + 12345u;
        float jitter = (hash & 0xFFFFFF) / float(0x1000000);

        return (stratum + jitter) / float(strataCount);
    }
};

float balanceHeuristic(float pdfA, float pdfB) {
    return pdfA / (pdfA + pdfB);
}

float powerHeuristic(float pdfA, float pdfB, float beta = 2.0f) {
    float pdfAPow = pow(pdfA, beta);
    float pdfBPow = pow(pdfB, beta);
    return pdfAPow / (pdfAPow + pdfBPow);
}

vec3 randomCosineDirection(const vec2& u) {
    float r = sqrt(u.x);
    float theta = 2.0f * pi<float>() * u.y;
    float x = r * cos(theta);
    float y = r * sin(theta);
    float z = sqrt(max(0.0f, 1.0f - u.x));
    return vec3(x, y, z);
}

float schlickFresnel(float cosine, float ior) {
    float r0 = (1.0f - ior) / (1.0f + ior);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * pow(1.0f - cosine, 5.0f);
}

vec3 ggxSample(const vec3& normal, float roughness, const vec2& u) {
    float a = roughness * roughness;
    float a2 = a * a;

    float phi = 2.0f * pi<float>() * u.x;
    float cosTheta = sqrt((1.0f - u.y) / (1.0f + (a2 - 1.0f) * u.y));
    float sinTheta = sqrt(1.0f - cosTheta * cosTheta);

    vec3 H = vec3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);

    vec3 up = abs(normal.z) < 0.999f ? vec3(0, 0, 1) : vec3(1, 0, 0);
    vec3 tangent = normalize(cross(up, normal));
    vec3 bitangent = cross(normal, tangent);

    return tangent * H.x + bitangent * H.y + normal * H.z;
}

float ggxPdf(const vec3& normal, const vec3& H, const vec3& wo, const vec3& wi, float roughness) {
    float NdotH = max(dot(normal, H), 0.0f);
    float VdotH = max(dot(wo, H), 0.0f);

    float a = roughness * roughness;
    float a2 = a * a;
    float denom = NdotH * NdotH * (a2 - 1.0f) + 1.0f;
    float D = a2 / (pi<float>() * denom * denom);

    return D * NdotH / (4.0f * VdotH);
}

vec3 evaluateBRDF(const HitInfo& hit, const vec3& wo, const vec3& wi, float& pdf) {
    const Material& mat = hit.material;
    pdf = 0.0f;

    float NdotL = dot(hit.normal, wi);
    float NdotV = dot(hit.normal, wo);

    if (NdotL <= 0.0f || NdotV <= 0.0f) return vec3(0.0f);

    vec3 result = vec3(0.0f);
    float totalPdf = 0.0f;

    float diffuseWeight = (1.0f - mat.metallic) * (1.0f - mat.transmission);
    if (diffuseWeight > 0.0f) {
        vec3 diffuse = mat.albedo / pi<float>();
        float diffusePdf = NdotL / pi<float>();
        result += diffuse * diffuseWeight;
        totalPdf += diffusePdf * diffuseWeight;
    }

    float specularWeight = 1.0f - mat.transmission;
    if (specularWeight > 0.0f) {
        vec3 H = normalize(wo + wi);
        float NdotH = max(dot(hit.normal, H), 0.0f);
        float VdotH = max(dot(wo, H), 0.0f);

        float a = mat.roughness * mat.roughness;
        float a2 = a * a;
        float denom = NdotH * NdotH * (a2 - 1.0f) + 1.0f;
        float D = a2 / (pi<float>() * denom * denom);

        float k = (mat.roughness + 1.0f) * (mat.roughness + 1.0f) / 8.0f;
        float G1L = NdotL / (NdotL * (1.0f - k) + k);
        float G1V = NdotV / (NdotV * (1.0f - k) + k);
        float G = G1L * G1V;

        vec3 F0 = mix(vec3(0.04f), mat.albedo, mat.metallic);
        vec3 F = F0 + (vec3(1.0f) - F0) * pow(1.0f - VdotH, 5.0f);

        vec3 specular = D * G * F / (4.0f * NdotL * NdotV);
        float specularPdf = ggxPdf(hit.normal, H, wo, wi, mat.roughness);

        result += specular * specularWeight;
        totalPdf += specularPdf * specularWeight;
    }

    pdf = totalPdf;
    return result;
}

vec3 sampleBRDF(const HitInfo& hit, const vec3& wo, vec3& wi, float& pdf, Sampler& sampler, float& materialComplexity) {
    const Material& mat = hit.material;
    materialComplexity = mat.getComplexity();

    float metallicWeight = mat.metallic;
    float transmissionWeight = mat.transmission * (1.0f - mat.metallic);
    float diffuseWeight = (1.0f - mat.metallic) * (1.0f - mat.transmission);
    float clearcoatWeight = mat.clearcoat;

    float totalWeight = metallicWeight + transmissionWeight + diffuseWeight + clearcoatWeight;
    if (totalWeight == 0.0f) return vec3(0.0f);

    float rand = sampler.get1D() * totalWeight;

    if (rand < diffuseWeight) {
        vec3 localWi = randomCosineDirection(sampler.get2D());
        vec3 up = abs(hit.normal.z) < 0.999f ? vec3(0, 0, 1) : vec3(1, 0, 0);
        vec3 tangent = normalize(cross(up, hit.normal));
        vec3 bitangent = cross(hit.normal, tangent);
        wi = normalize(tangent * localWi.x + bitangent * localWi.y + hit.normal * localWi.z);

        return evaluateBRDF(hit, wo, wi, pdf);
    }
    else if (rand < diffuseWeight + metallicWeight) {
        vec3 H = ggxSample(hit.normal, mat.roughness, sampler.get2D());
        wi = reflect(-wo, H);

        if (dot(wi, hit.normal) <= 0.0f) {
            pdf = 0.0f;
            return vec3(0.0f);
        }

        return evaluateBRDF(hit, wo, wi, pdf);
    }
    else if (rand < diffuseWeight + metallicWeight + transmissionWeight) {
        float cosTheta = dot(-wo, hit.normal);
        bool entering = cosTheta > 0.0f;
        float eta = entering ? (1.0f / mat.ior) : mat.ior;
        vec3 normal = entering ? hit.normal : -hit.normal;

        float fresnel = schlickFresnel(abs(cosTheta), eta);

        if (sampler.get1D() < fresnel || mat.roughness > 0.01f) {
            vec3 H = ggxSample(normal, mat.roughness, sampler.get2D());
            wi = reflect(-wo, H);
            pdf = fresnel;
            return vec3(fresnel) * (transmissionWeight / totalWeight);
        }
        else {
            vec3 refracted = refract(-wo, normal, eta);
            if (length(refracted) == 0.0f) {
                wi = reflect(-wo, normal);
                pdf = 1.0f;
                return vec3(1.0f) * (transmissionWeight / totalWeight);
            }
            wi = normalize(refracted);
            pdf = 1.0f - fresnel;
            return mat.albedo * (1.0f - fresnel) * (transmissionWeight / totalWeight);
        }
    }
    else {
        vec3 H = ggxSample(hit.normal, mat.clearcoatRoughness, sampler.get2D());
        wi = reflect(-wo, H);

        if (dot(wi, hit.normal) <= 0.0f) {
            pdf = 0.0f;
            return vec3(0.0f);
        }

        float F = schlickFresnel(dot(H, wo), 1.5f);
        pdf = F * mat.clearcoat;
        return vec3(F * mat.clearcoat) * (clearcoatWeight / totalWeight);
    }
}

vec3 evaluateDirectLighting(const HitInfo& hit, const vec3& wo, const Scene& scene, Sampler& sampler) {
    if (scene.getLights().empty()) return vec3(0.0f);

    vec3 result = vec3(0.0f);

    vec3 lightPos, lightNormal;
    float lightPdf;
    vec3 Le = scene.sampleLight(sampler.get1D(), lightPos, lightNormal, lightPdf);

    if (lightPdf > 0.0f) {
        vec3 wi = normalize(lightPos - hit.position);
        float distance = length(lightPos - hit.position);

        if (scene.visible(hit.position, lightPos)) {
            float bsdfPdf;
            vec3 f = evaluateBRDF(hit, wo, wi, bsdfPdf);

            if (bsdfPdf > 0.0f) {
                float cosThetaLight = max(-dot(lightNormal, wi), 0.0f);
                float geometryFactor = cosThetaLight / (distance * distance);

                float misWeight = powerHeuristic(lightPdf, bsdfPdf * geometryFactor);

                result = f * Le * geometryFactor * misWeight / lightPdf;
            }
        }
    }

    return result;
}

struct PathState {
    vec3 throughput = vec3(1.0f);
    vec3 radiance = vec3(0.0f);
    float accumulatedComplexity = 0.0f;
    int bounces = 0;
};

vec3 pathTrace(const Ray& ray, const Scene& scene, Sampler& sampler, int maxBounces, float& totalComplexity) {
    PathState state;
    Ray currentRay = ray;

    for (int bounce = 0; bounce < maxBounces; ++bounce) {
        HitInfo hit = scene.intersect(currentRay);

        if (!hit.hit) {
            state.radiance += state.throughput * vec3(0.1f, 0.2f, 0.3f);
            break;
        }

        vec3 wo = -currentRay.direction;

        if (hit.material.isLight() && bounce == 0) {
            state.radiance += state.throughput * hit.material.emissive;
        }

        state.radiance += state.throughput * evaluateDirectLighting(hit, wo, scene, sampler);

        vec3 wi;
        float pdf;
        float materialComplexity;

        vec3 f = sampleBRDF(hit, wo, wi, pdf, sampler, materialComplexity);

        if (pdf <= 0.0f || all(equal(f, vec3(0.0f)))) break;

        state.throughput *= f * abs(dot(wi, hit.normal)) / pdf;

        state.throughput = min(state.throughput, vec3(10.0f));

        state.accumulatedComplexity += materialComplexity;

        currentRay.origin = hit.position + hit.normal * 0.001f;
        currentRay.direction = wi;

        if (bounce > 3) {
            float p = max(state.throughput.x, max(state.throughput.y, state.throughput.z));
            if (sampler.get1D() > p) break;
            state.throughput /= p;
        }
    }

    totalComplexity = state.accumulatedComplexity;
    return state.radiance;
}

class AdaptiveRenderer {
    int width, height;
    int minSamples = 32;
    int maxSamples = 256;
    float complexityThreshold = 0.5f;

public:
    AdaptiveRenderer(int w, int h) : width(w), height(h) {}

    struct PixelData {
        vec3 color = vec3(0.0f);
        float variance = 0.0f;
        float complexity = 0.0f;
        int samples = 0;
    };

    void render(const Scene& scene, bool adaptive) {
        std::vector<PixelData> pixels(width * height);
        auto startTime = std::chrono::high_resolution_clock::now();

        #pragma omp parallel for schedule(dynamic)
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int idx = y * width + x;
                PixelData& pixel = pixels[idx];

                int targetSamples = adaptive ? minSamples : 64;
                vec3 colorSum = vec3(0.0f);
                vec3 colorSumSq = vec3(0.0f);
                float complexitySum = 0.0f;

                for (int s = 0; s < targetSamples; ++s) {
                    Sampler sampler(idx, s);
                    vec2 jitter = sampler.get2D();

                    float u = (x + jitter.x) / width;
                    float v = (y + jitter.y) / height;

                    vec3 origin(0, 0, 5);
                    vec3 direction = normalize(vec3(
                        (u - 0.5f) * 2.0f * float(width) / height,
                        (0.5f - v) * 2.0f,
                        -2.0f
                    ));

                    Ray ray{origin, direction};
                    float complexity;
                    vec3 color = pathTrace(ray, scene, sampler, 8, complexity);

                    colorSum += color;
                    colorSumSq += color * color;
                    complexitySum += complexity;
                    pixel.samples++;
                }

                pixel.color = colorSum / float(pixel.samples);
                vec3 variance = (colorSumSq / float(pixel.samples)) - (pixel.color * pixel.color);
                pixel.variance = (variance.x + variance.y + variance.z) / 3.0f;
                pixel.complexity = complexitySum / float(pixel.samples);

                if (adaptive && pixel.complexity > complexityThreshold) {
                    int additionalSamples = int((pixel.complexity - complexityThreshold) * (maxSamples - minSamples));

                    for (int s = 0; s < additionalSamples; ++s) {
                        Sampler sampler(idx, targetSamples + s);
                        vec2 jitter = sampler.get2D();

                        float u = (x + jitter.x) / width;
                        float v = (y + jitter.y) / height;

                        vec3 origin(0, 0, 5);
                        vec3 direction = normalize(vec3(
                            (u - 0.5f) * 2.0f * float(width) / height,
                            (0.5f - v) * 2.0f,
                            -2.0f
                        ));

                        Ray ray{origin, direction};
                        float complexity;
                        vec3 color = pathTrace(ray, scene, sampler, 8, complexity);

                        pixel.color = (pixel.color * float(pixel.samples) + color) / float(pixel.samples + 1);
                        pixel.samples++;
                    }
                }
            }
        }

        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();

        saveImage(pixels, adaptive ? "adaptive_render.png" : "uniform_render.png");
        saveSamplesMap(pixels, adaptive ? "adaptive_samples.png" : "uniform_samples.png");
        saveComplexityMap(pixels, adaptive ? "adaptive_complexity.png" : "uniform_complexity.png");
        saveVarianceMap(pixels, adaptive ? "adaptive_variance.png" : "uniform_variance.png");

        float avgSamples = 0.0f;
        for (const auto& p : pixels) avgSamples += p.samples;
        avgSamples /= pixels.size();

        std::cout << (adaptive ? "Adaptive" : "Uniform") << " rendering:" << std::endl;
        std::cout << "  Temps: " << duration << "ms" << std::endl;
        std::cout << "  Samples moyens: " << avgSamples << std::endl;
    }

private:
    void saveImage(const std::vector<PixelData>& pixels, const std::string& filename) {
        std::vector<unsigned char> image(width * height * 3);

        for (int i = 0; i < width * height; ++i) {
            vec3 color = clamp(pixels[i].color, 0.0f, 1.0f);
            color = pow(color, vec3(1.0f / 2.2f));

            image[i * 3 + 0] = static_cast<unsigned char>(color.r * 255);
            image[i * 3 + 1] = static_cast<unsigned char>(color.g * 255);
            image[i * 3 + 2] = static_cast<unsigned char>(color.b * 255);
        }

        stbi_write_png(filename.c_str(), width, height, 3, image.data(), width * 3);
    }

    void saveSamplesMap(const std::vector<PixelData>& pixels, const std::string& filename) {
        std::vector<unsigned char> image(width * height * 3);

        float maxSamplesInScene = 0.0f;
        for (const auto& p : pixels) {
            maxSamplesInScene = max(maxSamplesInScene, float(p.samples));
        }

        for (int i = 0; i < width * height; ++i) {
            float normalized = float(pixels[i].samples) / maxSamplesInScene;
            vec3 color = mix(vec3(0, 0, 1), vec3(1, 0, 0), normalized);

            image[i * 3 + 0] = static_cast<unsigned char>(color.r * 255);
            image[i * 3 + 1] = static_cast<unsigned char>(color.g * 255);
            image[i * 3 + 2] = static_cast<unsigned char>(color.b * 255);
        }

        stbi_write_png(filename.c_str(), width, height, 3, image.data(), width * 3);
    }

    void saveComplexityMap(const std::vector<PixelData>& pixels, const std::string& filename) {
        std::vector<unsigned char> image(width * height * 3);

        for (int i = 0; i < width * height; ++i) {
            float complexity = clamp(pixels[i].complexity, 0.0f, 1.0f);
            vec3 color = mix(vec3(0, 0.2f, 0.4f), vec3(1, 0.8f, 0), complexity);

            image[i * 3 + 0] = static_cast<unsigned char>(color.r * 255);
            image[i * 3 + 1] = static_cast<unsigned char>(color.g * 255);
            image[i * 3 + 2] = static_cast<unsigned char>(color.b * 255);
        }

        stbi_write_png(filename.c_str(), width, height, 3, image.data(), width * 3);
    }

    void saveVarianceMap(const std::vector<PixelData>& pixels, const std::string& filename) {
        std::vector<unsigned char> image(width * height * 3);

        float maxVariance = 0.0f;
        for (const auto& p : pixels) {
            maxVariance = max(maxVariance, p.variance);
        }

        for (int i = 0; i < width * height; ++i) {
            float normalized = pixels[i].variance / (maxVariance + 0.0001f);
            vec3 color = vec3(normalized);

            image[i * 3 + 0] = static_cast<unsigned char>(color.r * 255);
            image[i * 3 + 1] = static_cast<unsigned char>(color.g * 255);
            image[i * 3 + 2] = static_cast<unsigned char>(color.b * 255);
        }

        stbi_write_png(filename.c_str(), width, height, 3, image.data(), width * 3);
    }
};

int main() {
    Scene scene;

    Material matDiffuse;
    matDiffuse.albedo = vec3(0.8f, 0.3f, 0.3f);
    matDiffuse.roughness = 0.8f;

    Material matMetal;
    matMetal.albedo = vec3(0.9f, 0.9f, 0.9f);
    matMetal.metallic = 1.0f;
    matMetal.roughness = 0.1f;

    Material matGlass;
    matGlass.albedo = vec3(0.95f);
    matGlass.transmission = 1.0f;
    matGlass.ior = 1.5f;
    matGlass.roughness = 0.0f;

    Material matComplex;
    matComplex.albedo = vec3(0.2f, 0.4f, 0.8f);
    matComplex.metallic = 0.3f;
    matComplex.roughness = 0.2f;
    matComplex.clearcoat = 0.8f;
    matComplex.subsurfaceWeight = 0.3f;

    Material matLight;
    matLight.emissive = vec3(15.0f);

    Material matSmallLight;
    matSmallLight.emissive = vec3(50.0f);

    scene.add(std::make_unique<Sphere>(vec3(-2, 0, 0), 1.0f, matDiffuse));
    scene.add(std::make_unique<Sphere>(vec3(0, 0, 0), 1.0f, matGlass));
    scene.add(std::make_unique<Sphere>(vec3(2, 0, 0), 1.0f, matMetal));
    scene.add(std::make_unique<Sphere>(vec3(0, -101, 0), 100.0f, matDiffuse));
    scene.add(std::make_unique<Sphere>(vec3(-1, 2, 1), 0.8f, matComplex));
    scene.add(std::make_unique<Sphere>(vec3(0, 5, 0), 1.0f, matLight));
    scene.add(std::make_unique<Sphere>(vec3(3, 2.5f, -1), 0.3f, matSmallLight));

    AdaptiveRenderer renderer(800, 600);

    std::cout << "Rendu uniforme..." << std::endl;
    renderer.render(scene, false);

    std::cout << "\nRendu adaptatif..." << std::endl;
    renderer.render(scene, true);

    return 0;
}