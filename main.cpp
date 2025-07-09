#include <glm/glm.hpp>
#include <glm/gtc/random.hpp>
#include <glm/gtc/constants.hpp>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#include <vector>
#include <memory>
#include <random>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <omp.h>

using namespace glm;

struct Config {
    int width = 800;
    int height = 600;
    int minSamples = 32;
    int maxSamples = 256;
    int uniformSamples = 64;
    int groundTruthSamples = 1000;
    int maxBounces = 8;
    float complexityThreshold = 0.5f;
    bool enableNEE = true;
    bool enableMIS = true;
    bool enableRussianRoulette = true;
    bool enableAdaptive = true;
    std::string outputPrefix = "output";
};

void printHelp() {
    std::cout << R"(
Path Tracer Adaptatif - Système Expérimental

Usage: pathtracer [options]

Options de rendu:
  --width <n>              Largeur de l'image (défaut: 800)
  --height <n>             Hauteur de l'image (défaut: 600)
  --max-bounces <n>        Nombre max de rebonds (défaut: 8)

Options d'échantillonnage:
  --min-samples <n>        Samples minimum (adaptatif) (défaut: 32)
  --max-samples <n>        Samples maximum (adaptatif) (défaut: 256)
  --uniform-samples <n>    Samples pour mode uniforme (défaut: 64)
  --gt-samples <n>         Samples pour ground truth (défaut: 1000)
  --complexity-threshold <f> Seuil de complexité (défaut: 0.5)

Features on/off:
  --no-nee                 Désactiver Next Event Estimation
  --no-mis                 Désactiver Multiple Importance Sampling
  --no-rr                  Désactiver Russian Roulette
  --no-adaptive            Désactiver échantillonnage adaptatif

Sortie:
  --output <prefix>        Préfixe des fichiers de sortie (défaut: "output")
  --help                   Afficher cette aide

Exemples:
  # Rendu rapide de test
  pathtracer --width 400 --height 300 --uniform-samples 32

  # Test sans NEE
  pathtracer --no-nee --uniform-samples 128 --output test_no_nee

  # Ground truth haute qualité
  pathtracer --uniform-samples 1000 --output ground_truth

  # Échantillonnage adaptatif
  pathtracer --min-samples 16 --max-samples 256 --complexity-threshold 0.4
)";
}

Config parseArgs(int argc, char* argv[]) {
    Config config;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--help") == 0) {
            printHelp();
            exit(0);
        } else if (strcmp(argv[i], "--width") == 0 && i + 1 < argc) {
            config.width = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--height") == 0 && i + 1 < argc) {
            config.height = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--min-samples") == 0 && i + 1 < argc) {
            config.minSamples = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--max-samples") == 0 && i + 1 < argc) {
            config.maxSamples = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--uniform-samples") == 0 && i + 1 < argc) {
            config.uniformSamples = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--gt-samples") == 0 && i + 1 < argc) {
            config.groundTruthSamples = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--max-bounces") == 0 && i + 1 < argc) {
            config.maxBounces = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--complexity-threshold") == 0 && i + 1 < argc) {
            config.complexityThreshold = std::stof(argv[++i]);
        } else if (strcmp(argv[i], "--no-nee") == 0) {
            config.enableNEE = false;
        } else if (strcmp(argv[i], "--no-mis") == 0) {
            config.enableMIS = false;
        } else if (strcmp(argv[i], "--no-rr") == 0) {
            config.enableRussianRoulette = false;
        } else if (strcmp(argv[i], "--no-adaptive") == 0) {
            config.enableAdaptive = false;
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            config.outputPrefix = argv[++i];
        }
    }
    return config;
}

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

class Plane : public Shape {
    vec3 point;
    vec3 normal;
    Material material;
    float width, height;

public:
    Plane(const vec3& p, const vec3& n, float w, float h, const Material& m)
        : point(p), normal(normalize(n)), width(w), height(h), material(m) {}

    HitInfo intersect(const Ray& ray) const override {
        HitInfo info;
        float denom = dot(normal, ray.direction);

        if (abs(denom) > 0.0001f) {
            float t = dot(point - ray.origin, normal) / denom;
            if (t >= 0.001f) {
                vec3 hitPoint = ray.origin + t * ray.direction;
                vec3 localPos = hitPoint - point;

                vec3 u = normalize(cross(normal, vec3(0, 1, 0)));
                if (length(u) < 0.001f) u = normalize(cross(normal, vec3(1, 0, 0)));
                vec3 v = cross(normal, u);

                float localU = dot(localPos, u);
                float localV = dot(localPos, v);

                if (abs(localU) <= width * 0.5f && abs(localV) <= height * 0.5f) {
                    info.hit = true;
                    info.t = t;
                    info.position = hitPoint;
                    info.normal = normal;
                    info.material = material;
                }
            }
        }

        return info;
    }

    vec3 sampleSurface(float u1, float u2, vec3& n, float& pdf) const override {
        vec3 u = normalize(cross(normal, vec3(0, 1, 0)));
        if (length(u) < 0.001f) u = normalize(cross(normal, vec3(1, 0, 0)));
        vec3 v = cross(normal, u);

        float localU = (u1 - 0.5f) * width;
        float localV = (u2 - 0.5f) * height;

        n = normal;
        pdf = 1.0f / area();

        return point + localU * u + localV * v;
    }

    float area() const override {
        return width * height;
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
        Material mat;
        if (auto* sphere = dynamic_cast<Sphere*>(shape.get())) {
            mat = sphere->getMaterial();
        } else if (auto* plane = dynamic_cast<Plane*>(shape.get())) {
            mat = plane->getMaterial();
        }

        if (mat.isLight()) {
            LightInfo light;
            light.shape = shape.get();
            light.emission = mat.emissive;
            light.area = shape->area();
            lights.push_back(light);
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

vec3 evaluateDirectLighting(const HitInfo& hit, const vec3& wo, const Scene& scene, Sampler& sampler, bool enableNEE, bool enableMIS) {
    if (!enableNEE || scene.getLights().empty()) return vec3(0.0f);

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

                float misWeight = 1.0f;
                if (enableMIS) {
                    misWeight = powerHeuristic(lightPdf, bsdfPdf * geometryFactor);
                }

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

vec3 pathTrace(const Ray& ray, const Scene& scene, Sampler& sampler, const Config& config, float& totalComplexity) {
    PathState state;
    Ray currentRay = ray;

    for (int bounce = 0; bounce < config.maxBounces; ++bounce) {
        HitInfo hit = scene.intersect(currentRay);

        if (!hit.hit) {
            state.radiance += state.throughput * vec3(0.1f, 0.2f, 0.3f);
            break;
        }

        vec3 wo = -currentRay.direction;

        if (hit.material.isLight() && bounce == 0) {
            state.radiance += state.throughput * hit.material.emissive;
        }

        state.radiance += state.throughput * evaluateDirectLighting(hit, wo, scene, sampler, config.enableNEE, config.enableMIS);

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

        if (config.enableRussianRoulette && bounce > 3) {
            float p = max(state.throughput.x, max(state.throughput.y, state.throughput.z));
            if (sampler.get1D() > p) break;
            state.throughput /= p;
        }
    }

    totalComplexity = state.accumulatedComplexity;
    return state.radiance;
}

class Renderer {
    int width, height;
    Config config;

public:
    Renderer(int w, int h, const Config& cfg) : width(w), height(h), config(cfg) {}

    struct PixelData {
        vec3 color = vec3(0.0f);
        float variance = 0.0f;
        float complexity = 0.0f;
        int samples = 0;
    };

    struct RenderResult {
        std::vector<PixelData> pixels;
        float renderTime = 0.0f;
        float avgSamples = 0.0f;
        float avgVariance = 0.0f;
        float mse = 0.0f;
        std::string name;
    };

    RenderResult render(const Scene& scene, const std::string& name, int samplesPerPixel, bool adaptive) {
        RenderResult result;
        result.name = name;
        result.pixels.resize(width * height);

        auto startTime = std::chrono::high_resolution_clock::now();

        #pragma omp parallel for schedule(dynamic)
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int idx = y * width + x;
                PixelData& pixel = result.pixels[idx];

                int targetSamples = adaptive ? config.minSamples : samplesPerPixel;
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
                    vec3 color = pathTrace(ray, scene, sampler, config, complexity);

                    colorSum += color;
                    colorSumSq += color * color;
                    complexitySum += complexity;
                    pixel.samples++;
                }

                pixel.color = colorSum / float(pixel.samples);
                vec3 variance = (colorSumSq / float(pixel.samples)) - (pixel.color * pixel.color);
                pixel.variance = (variance.x + variance.y + variance.z) / 3.0f;
                pixel.complexity = complexitySum / float(pixel.samples);

                if (adaptive && config.enableAdaptive && pixel.complexity > config.complexityThreshold) {
                    int additionalSamples = int((pixel.complexity - config.complexityThreshold) * (config.maxSamples - config.minSamples));

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
                        vec3 color = pathTrace(ray, scene, sampler, config, complexity);

                        pixel.color = (pixel.color * float(pixel.samples) + color) / float(pixel.samples + 1);
                        pixel.samples++;
                    }
                }
            }
        }

        auto endTime = std::chrono::high_resolution_clock::now();
        result.renderTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count() / 1000.0f;

        for (const auto& p : result.pixels) {
            result.avgSamples += p.samples;
            result.avgVariance += p.variance;
        }
        result.avgSamples /= result.pixels.size();
        result.avgVariance /= result.pixels.size();

        return result;
    }

    float computeMSE(const RenderResult& test, const RenderResult& reference) {
        float mse = 0.0f;
        for (int i = 0; i < width * height; ++i) {
            vec3 diff = test.pixels[i].color - reference.pixels[i].color;
            mse += dot(diff, diff);
        }
        return mse / (width * height * 3.0f);
    }

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

    void saveMetrics(const std::vector<RenderResult>& results, const RenderResult& groundTruth) {
        std::ofstream file(config.outputPrefix + "_metrics.json");
        file << "{\n";
        file << "  \"ground_truth\": {\n";
        file << "    \"samples\": " << groundTruth.avgSamples << ",\n";
        file << "    \"time\": " << groundTruth.renderTime << "\n";
        file << "  },\n";
        file << "  \"tests\": [\n";

        for (size_t i = 0; i < results.size(); ++i) {
            const auto& result = results[i];
            float mse = computeMSE(result, groundTruth);

            file << "    {\n";
            file << "      \"name\": \"" << result.name << "\",\n";
            file << "      \"mse\": " << mse << ",\n";
            file << "      \"variance\": " << result.avgVariance << ",\n";
            file << "      \"time\": " << result.renderTime << ",\n";
            file << "      \"avg_samples\": " << result.avgSamples << "\n";
            file << "    }";

            if (i < results.size() - 1) file << ",";
            file << "\n";
        }

        file << "  ]\n";
        file << "}\n";
    }
};

Scene createTestScene() {
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

    Material matGreenWall;
    matGreenWall.albedo = vec3(0.1f, 0.8f, 0.1f);
    matGreenWall.roughness = 0.9f;

    Material matBlueWall;
    matBlueWall.albedo = vec3(0.1f, 0.1f, 0.8f);
    matBlueWall.roughness = 0.9f;

    scene.add(std::make_unique<Sphere>(vec3(-2, 0, 0), 1.0f, matDiffuse));
    scene.add(std::make_unique<Sphere>(vec3(0, 0, 0), 1.0f, matGlass));
    scene.add(std::make_unique<Sphere>(vec3(2, 0, 0), 1.0f, matMetal));
    scene.add(std::make_unique<Sphere>(vec3(-1, 2, 1), 0.8f, matComplex));
    scene.add(std::make_unique<Sphere>(vec3(0, 5, 0), 1.0f, matLight));
    scene.add(std::make_unique<Sphere>(vec3(3, 2.5f, -1), 0.3f, matSmallLight));

    scene.add(std::make_unique<Plane>(vec3(0, -1, 0), vec3(0, 1, 0), 12.0f, 12.0f, matDiffuse));
    scene.add(std::make_unique<Plane>(vec3(0, 6, 0), vec3(0, -1, 0), 12.0f, 12.0f, matDiffuse));
    scene.add(std::make_unique<Plane>(vec3(-6, 2.5f, 0), vec3(1, 0, 0), 12.0f, 7.0f, matGreenWall));
    scene.add(std::make_unique<Plane>(vec3(6, 2.5f, 0), vec3(-1, 0, 0), 12.0f, 7.0f, matBlueWall));
    scene.add(std::make_unique<Plane>(vec3(0, 2.5f, -6), vec3(0, 0, 1), 12.0f, 7.0f, matDiffuse));

    return scene;
}

void generateHTMLReport(const Config& config) {
    std::ofstream html(config.outputPrefix + "_report.html");

    html << R"(<!DOCTYPE html>
<html>
<head>
    <title>Path Tracer Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .comparison { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
        img { width: 100%; height: auto; }
        .metric { background: #f0f0f0; padding: 10px; margin: 10px 0; }
        h2 { color: #333; }
        .chart { width: 100%; height: 400px; margin: 20px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Path Tracer Experimental Results</h1>

        <h2>Ground Truth Reference</h2>
        <img src=")" + config.outputPrefix + R"(_ground_truth.png" alt="Ground Truth">

        <h2>Comparisons</h2>
        <div class="grid">
            <div>
                <h3>No NEE</h3>
                <img src=")" + config.outputPrefix + R"(_no_nee.png">
            </div>
            <div>
                <h3>With NEE</h3>
                <img src=")" + config.outputPrefix + R"(_with_nee.png">
            </div>
            <div>
                <h3>NEE + MIS</h3>
                <img src=")" + config.outputPrefix + R"(_nee_mis.png">
            </div>
            <div>
                <h3>Full Features</h3>
                <img src=")" + config.outputPrefix + R"(_full.png">
            </div>
        </div>

        <h2>Adaptive vs Uniform</h2>
        <div class="comparison">
            <div>
                <h3>Uniform Sampling</h3>
                <img src=")" + config.outputPrefix + R"(_uniform.png">
                <img src=")" + config.outputPrefix + R"(_uniform_samples.png">
            </div>
            <div>
                <h3>Adaptive Sampling</h3>
                <img src=")" + config.outputPrefix + R"(_adaptive.png">
                <img src=")" + config.outputPrefix + R"(_adaptive_samples.png">
            </div>
        </div>

        <h2>Analysis Maps</h2>
        <div class="grid">
            <div>
                <h3>Complexity Map</h3>
                <img src=")" + config.outputPrefix + R"(_adaptive_complexity.png">
            </div>
            <div>
                <h3>Variance Map</h3>
                <img src=")" + config.outputPrefix + R"(_adaptive_variance.png">
            </div>
        </div>

        <div id="mseChart" class="chart"></div>
        <div id="varianceChart" class="chart"></div>
        <div id="timeChart" class="chart"></div>

        <script>
            fetch(')" + config.outputPrefix + R"(_metrics.json')
                .then(response => response.json())
                .then(data => {
                    const names = data.tests.map(t => t.name);
                    const mse = data.tests.map(t => t.mse);
                    const variance = data.tests.map(t => t.variance);
                    const time = data.tests.map(t => t.time);
                    const samples = data.tests.map(t => t.avg_samples);

                    Plotly.newPlot('mseChart', [{
                        x: names,
                        y: mse,
                        type: 'bar',
                        name: 'MSE'
                    }], {
                        title: 'Mean Squared Error vs Ground Truth',
                        yaxis: { title: 'MSE' }
                    });

                    Plotly.newPlot('varianceChart', [{
                        x: samples,
                        y: variance,
                        mode: 'markers+lines',
                        type: 'scatter',
                        text: names,
                        textposition: 'top center'
                    }], {
                        title: 'Variance vs Sample Count',
                        xaxis: { title: 'Average Samples per Pixel' },
                        yaxis: { title: 'Average Variance' }
                    });

                    Plotly.newPlot('timeChart', [{
                        x: time,
                        y: mse,
                        mode: 'markers+text',
                        type: 'scatter',
                        text: names,
                        textposition: 'top center',
                        marker: { size: 10 }
                    }], {
                        title: 'Quality vs Render Time',
                        xaxis: { title: 'Render Time (seconds)' },
                        yaxis: { title: 'MSE', type: 'log' }
                    });
                });
        </script>
    </div>
</body>
</html>)";
}

int main(int argc, char* argv[]) {
    Config config = parseArgs(argc, argv);

    std::cout << "Configuration:" << std::endl;
    std::cout << "  Resolution: " << config.width << "x" << config.height << std::endl;
    std::cout << "  Samples: " << config.minSamples << "-" << config.maxSamples << " (adaptive)" << std::endl;
    std::cout << "  Max bounces: " << config.maxBounces << std::endl;
    std::cout << "  Features: NEE=" << config.enableNEE << " MIS=" << config.enableMIS
              << " RR=" << config.enableRussianRoulette << " Adaptive=" << config.enableAdaptive << std::endl;

    Scene scene = createTestScene();
    Renderer renderer(config.width, config.height, config);

    std::vector<Renderer::RenderResult> results;

    std::cout << "\n1. Generating ground truth (" << config.groundTruthSamples << " spp)..." << std::endl;
    auto groundTruth = renderer.render(scene, "ground_truth", config.groundTruthSamples, false);
    renderer.saveImage(groundTruth.pixels, config.outputPrefix + "_ground_truth.png");

    std::cout << "\n2. Testing without NEE..." << std::endl;
    config.enableNEE = false;
    config.enableMIS = false;
    auto noNEE = renderer.render(scene, "no_nee", config.uniformSamples, false);
    renderer.saveImage(noNEE.pixels, config.outputPrefix + "_no_nee.png");
    results.push_back(noNEE);

    std::cout << "\n3. Testing with NEE only..." << std::endl;
    config.enableNEE = true;
    config.enableMIS = false;
    auto withNEE = renderer.render(scene, "with_nee", config.uniformSamples, false);
    renderer.saveImage(withNEE.pixels, config.outputPrefix + "_with_nee.png");
    results.push_back(withNEE);

    std::cout << "\n4. Testing with NEE + MIS..." << std::endl;
    config.enableNEE = true;
    config.enableMIS = true;
    auto neeMIS = renderer.render(scene, "nee_mis", config.uniformSamples, false);
    renderer.saveImage(neeMIS.pixels, config.outputPrefix + "_nee_mis.png");
    results.push_back(neeMIS);

    std::cout << "\n5. Testing full features..." << std::endl;
    config.enableRussianRoulette = true;
    auto full = renderer.render(scene, "full_features", config.uniformSamples, false);
    renderer.saveImage(full.pixels, config.outputPrefix + "_full.png");
    results.push_back(full);

    std::cout << "\n6. Testing uniform sampling..." << std::endl;
    auto uniform = renderer.render(scene, "uniform", config.uniformSamples, false);
    renderer.saveImage(uniform.pixels, config.outputPrefix + "_uniform.png");
    renderer.saveSamplesMap(uniform.pixels, config.outputPrefix + "_uniform_samples.png");
    renderer.saveVarianceMap(uniform.pixels, config.outputPrefix + "_uniform_variance.png");
    results.push_back(uniform);

    std::cout << "\n7. Testing adaptive sampling..." << std::endl;
    config.enableAdaptive = true;
    auto adaptive = renderer.render(scene, "adaptive", config.uniformSamples, true);
    renderer.saveImage(adaptive.pixels, config.outputPrefix + "_adaptive.png");
    renderer.saveSamplesMap(adaptive.pixels, config.outputPrefix + "_adaptive_samples.png");
    renderer.saveComplexityMap(adaptive.pixels, config.outputPrefix + "_adaptive_complexity.png");
    renderer.saveVarianceMap(adaptive.pixels, config.outputPrefix + "_adaptive_variance.png");
    results.push_back(adaptive);

    renderer.saveMetrics(results, groundTruth);
    generateHTMLReport(config);

    std::cout << "\nRendu terminé. Ouvrez " << config.outputPrefix << "_report.html pour voir les résultats." << std::endl;

    return 0;
}