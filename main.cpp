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
};

class Scene {
    std::vector<std::unique_ptr<Shape>> shapes;

public:
    void add(std::unique_ptr<Shape> shape) {
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
};

vec3 randomInUnitSphere(std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    vec3 p;
    do {
        p = vec3(dist(rng), dist(rng), dist(rng));
    } while (length(p) >= 1.0f);
    return p;
}

vec3 randomCosineDirection(std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r1 = dist(rng);
    float r2 = dist(rng);
    float z = sqrt(1.0f - r2);
    float phi = 2.0f * pi<float>() * r1;
    float x = cos(phi) * sqrt(r2);
    float y = sin(phi) * sqrt(r2);
    return vec3(x, y, z);
}

float schlickFresnel(float cosine, float ior) {
    float r0 = (1.0f - ior) / (1.0f + ior);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * pow(1.0f - cosine, 5.0f);
}

vec3 ggxSample(const vec3& normal, float roughness, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float a = roughness * roughness;
    float a2 = a * a;
    float r1 = dist(rng);
    float r2 = dist(rng);

    float phi = 2.0f * pi<float>() * r1;
    float cosTheta = sqrt((1.0f - r2) / (1.0f + (a2 - 1.0f) * r2));
    float sinTheta = sqrt(1.0f - cosTheta * cosTheta);

    vec3 H = vec3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);

    vec3 up = abs(normal.z) < 0.999f ? vec3(0, 0, 1) : vec3(1, 0, 0);
    vec3 tangent = normalize(cross(up, normal));
    vec3 bitangent = cross(normal, tangent);

    return tangent * H.x + bitangent * H.y + normal * H.z;
}

struct PathState {
    vec3 throughput = vec3(1.0f);
    vec3 radiance = vec3(0.0f);
    float accumulatedComplexity = 0.0f;
    int bounces = 0;
};

vec3 sampleBRDF(const HitInfo& hit, const vec3& wo, vec3& wi, float& pdf, std::mt19937& rng, float& materialComplexity) {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    const Material& mat = hit.material;
    materialComplexity = mat.getComplexity();

    float metallicWeight = mat.metallic;
    float transmissionWeight = mat.transmission * (1.0f - mat.metallic);
    float diffuseWeight = (1.0f - mat.metallic) * (1.0f - mat.transmission);
    float clearcoatWeight = mat.clearcoat;

    float totalWeight = metallicWeight + transmissionWeight + diffuseWeight + clearcoatWeight;
    if (totalWeight == 0.0f) return vec3(0.0f);

    float rand = dist(rng) * totalWeight;

    if (rand < diffuseWeight) {
        vec3 localWi = randomCosineDirection(rng);
        vec3 up = abs(hit.normal.z) < 0.999f ? vec3(0, 0, 1) : vec3(1, 0, 0);
        vec3 tangent = normalize(cross(up, hit.normal));
        vec3 bitangent = cross(hit.normal, tangent);
        wi = normalize(tangent * localWi.x + bitangent * localWi.y + hit.normal * localWi.z);

        pdf = dot(hit.normal, wi) / pi<float>();
        vec3 f = mat.albedo / pi<float>();

        if (mat.subsurfaceWeight > 0.0f) {
            f = mix(f, mat.albedo * exp(-length(hit.position) * 0.1f), mat.subsurfaceWeight);
        }

        return f * (diffuseWeight / totalWeight);
    }
    else if (rand < diffuseWeight + metallicWeight) {
        vec3 H = ggxSample(hit.normal, mat.roughness, rng);
        wi = reflect(-wo, H);

        if (dot(wi, hit.normal) <= 0.0f) {
            pdf = 0.0f;
            return vec3(0.0f);
        }

        float NdotL = dot(hit.normal, wi);
        float NdotV = dot(hit.normal, wo);
        float NdotH = dot(hit.normal, H);
        float VdotH = dot(wo, H);

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

        pdf = D * NdotH / (4.0f * VdotH);
        return D * G * F / (4.0f * NdotL * NdotV) * (metallicWeight / totalWeight);
    }
    else if (rand < diffuseWeight + metallicWeight + transmissionWeight) {
        float cosTheta = dot(-wo, hit.normal);
        bool entering = cosTheta > 0.0f;
        float eta = entering ? (1.0f / mat.ior) : mat.ior;
        vec3 normal = entering ? hit.normal : -hit.normal;

        float fresnel = schlickFresnel(abs(cosTheta), eta);

        if (dist(rng) < fresnel || mat.roughness > 0.01f) {
            vec3 H = ggxSample(normal, mat.roughness, rng);
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
        vec3 H = ggxSample(hit.normal, mat.clearcoatRoughness, rng);
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

vec3 pathTrace(const Ray& ray, const Scene& scene, std::mt19937& rng, int maxBounces, float& totalComplexity) {
    PathState state;
    Ray currentRay = ray;

    for (int bounce = 0; bounce < maxBounces; ++bounce) {
        HitInfo hit = scene.intersect(currentRay);

        if (!hit.hit) {
            state.radiance += state.throughput * vec3(0.1f, 0.2f, 0.3f);
            break;
        }

        state.radiance += state.throughput * hit.material.emissive;

        vec3 wo = -currentRay.direction;
        vec3 wi;
        float pdf;
        float materialComplexity;

        vec3 f = sampleBRDF(hit, wo, wi, pdf, rng, materialComplexity);

        if (pdf <= 0.0f || all(equal(f, vec3(0.0f)))) break;

        state.throughput *= f * abs(dot(wi, hit.normal)) / pdf;
        state.accumulatedComplexity += materialComplexity;

        currentRay.origin = hit.position + hit.normal * 0.001f;
        currentRay.direction = wi;

        if (bounce > 3) {
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            float p = max(state.throughput.x, max(state.throughput.y, state.throughput.z));
            if (dist(rng) > p) break;
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
            std::mt19937 rng(y * width + omp_get_thread_num());

            for (int x = 0; x < width; ++x) {
                int idx = y * width + x;
                PixelData& pixel = pixels[idx];

                int targetSamples = adaptive ? minSamples : 64;
                vec3 colorSum = vec3(0.0f);
                vec3 colorSumSq = vec3(0.0f);
                float complexitySum = 0.0f;

                for (int s = 0; s < targetSamples; ++s) {
                    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
                    float u = (x + dist(rng)) / width;
                    float v = (y + dist(rng)) / height;

                    vec3 origin(0, 0, 5);
                    vec3 direction = normalize(vec3(
                        (u - 0.5f) * 2.0f * float(width) / height,
                        (0.5f - v) * 2.0f,
                        -2.0f
                    ));

                    Ray ray{origin, direction};
                    float complexity;
                    vec3 color = pathTrace(ray, scene, rng, 8, complexity);

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
                        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
                        float u = (x + dist(rng)) / width;
                        float v = (y + dist(rng)) / height;

                        vec3 origin(0, 0, 5);
                        vec3 direction = normalize(vec3(
                            (u - 0.5f) * 2.0f * float(width) / height,
                            (0.5f - v) * 2.0f,
                            -2.0f
                        ));

                        Ray ray{origin, direction};
                        float complexity;
                        vec3 color = pathTrace(ray, scene, rng, 8, complexity);

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
    matLight.emissive = vec3(10.0f);

    scene.add(std::make_unique<Sphere>(vec3(-2, 0, 0), 1.0f, matDiffuse));
    scene.add(std::make_unique<Sphere>(vec3(0, 0, 0), 1.0f, matGlass));
    scene.add(std::make_unique<Sphere>(vec3(2, 0, 0), 1.0f, matMetal));
    scene.add(std::make_unique<Sphere>(vec3(0, -101, 0), 100.0f, matDiffuse));
    scene.add(std::make_unique<Sphere>(vec3(-1, 2, 1), 0.8f, matComplex));
    scene.add(std::make_unique<Sphere>(vec3(0, 5, 0), 1.0f, matLight));

    AdaptiveRenderer renderer(800, 600);

    std::cout << "Rendu uniforme..." << std::endl;
    renderer.render(scene, false);

    std::cout << "\nRendu adaptatif..." << std::endl;
    renderer.render(scene, true);

    return 0;
}