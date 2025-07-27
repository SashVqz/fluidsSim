#include "SphFluidModel.h"
#include "sph_kernels.h"
#include "utils/Logger.h"
#include <random>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>

SphFluidModel::SphFluidModel(int numParticles, const glm::vec3& boxMin, const glm::vec3& boxMax)
    : numParticles(numParticles), boxMin(boxMin), boxMax(boxMax),
      d_positions(nullptr), d_velocities(nullptr), d_forces(nullptr),
      d_densities(nullptr), d_pressures(nullptr),
      pxPhysics(nullptr), pxScene(nullptr), pxCudaContextManager(nullptr)
{
    // SPH parameters, like water is a sph fluid 
    particleMass = 0.02f;           
    restDensity = 1000.0f;          
    gasConstant = 3000.0f;          
    viscosity = 3.5f;               
    smoothingRadius = 0.0457f;      
    timeStep = 0.0008f;             
    dampingFactor = 0.99f;          
    
    spacing = pow(particleMass / restDensity, 1.0f/3.0f);
    
    hostParticlePositions.resize(numParticles);
    hostParticleRenderData.resize(numParticles * 6);
    
    Logger::info("SPH Fluid Model created with " + std::to_string(numParticles) + " particles");
    printSimulationInfo();
}

SphFluidModel::~SphFluidModel() {
    shutdown();
}

bool SphFluidModel::initialize(physx::PxPhysics* physics, physx::PxScene* scene, physx::PxCudaContextManager* cudaContextManager) {
    Logger::info("Initializing SPH Fluid Model with CUDA kernels");
    
    pxPhysics = physics;
    pxScene = scene;
    pxCudaContextManager = cudaContextManager;
    
    try {
        if (!validateParameters()) {
            Logger::error("SPH parameters validation failed");
            return false;
        }
        
        initializeParticles();
        initializeCudaBuffers();
        
        // Configurar constantes CUDA
        setupCudaConstants();
        
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            Logger::error("CUDA initialization error: " + std::string(cudaGetErrorString(error)));
            return false;
        }
        
        Logger::info("SPH Fluid Model with CUDA kernels initialized successfully");
        return true;
        
    } catch (const std::exception& e) {
        Logger::error("Failed to initialize SPH Fluid Model: " + std::string(e.what()));
        return false;
    }
}

void SphFluidModel::setupCudaConstants() {
    Logger::info("Setting up CUDA constants...");
    
    // cuda need float3 for gravity and box bounds
    float3 gravity = make_float3(0.0f, -9.81f, 0.0f);
    float3 boxMinCuda = make_float3(boxMin.x, boxMin.y, boxMin.z);
    float3 boxMaxCuda = make_float3(boxMax.x, boxMax.y, boxMax.z);
    
    setup_sph_constants(
        smoothingRadius,
        particleMass,
        restDensity,
        gasConstant,
        viscosity,
        dampingFactor,
        gravity,
        boxMinCuda,
        boxMaxCuda
    );
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error("Failed to setup CUDA constants: " + std::string(cudaGetErrorString(error)));
    }
}

void SphFluidModel::initializeParticles() {
    Logger::info("Initializing particle positions...");
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    glm::vec3 initMin = boxMin + glm::vec3(0.2f);
    glm::vec3 initMax = boxMax - glm::vec3(0.2f);
    initMax.y = boxMin.y + (boxMax.y - boxMin.y) * 0.4f;
    
    glm::vec3 region = initMax - initMin;
    int particlesX = static_cast<int>(region.x / spacing) + 1;
    int particlesY = static_cast<int>(region.y / spacing) + 1;
    int particlesZ = static_cast<int>(region.z / spacing) + 1;
    
    int totalGridParticles = particlesX * particlesY * particlesZ;
    
    if (totalGridParticles < numParticles) {
        Logger::warn("Grid arrangement can only fit " + std::to_string(totalGridParticles) + " particles, using random positions for the rest");
    }
    
    int particleIndex = 0;
    for (int x = 0; x < particlesX && particleIndex < numParticles; ++x) {
        for (int y = 0; y < particlesY && particleIndex < numParticles; ++y) {
            for (int z = 0; z < particlesZ && particleIndex < numParticles; ++z) {
                glm::vec3 gridPos = initMin + glm::vec3(x * spacing, y * spacing, z * spacing);
                
                glm::vec3 perturbation = glm::vec3(
                    (dis(gen) - 0.5f) * spacing * 0.1f,
                    (dis(gen) - 0.5f) * spacing * 0.1f,
                    (dis(gen) - 0.5f) * spacing * 0.1f
                );
                
                hostParticlePositions[particleIndex] = gridPos + perturbation;
                hostParticlePositions[particleIndex] = glm::clamp(
                    hostParticlePositions[particleIndex], 
                    initMin, 
                    initMax
                );
                
                particleIndex++;
            }
        }
    }
    
    for (int i = particleIndex; i < numParticles; ++i) {
        hostParticlePositions[i] = glm::vec3(
            initMin.x + dis(gen) * region.x,
            initMin.y + dis(gen) * region.y,
            initMin.z + dis(gen) * region.z
        );
    }
    
    Logger::info("Initialized " + std::to_string(numParticles) + " particles");
}

void SphFluidModel::initializeCudaBuffers() {
    Logger::info("Initializing CUDA buffers");
    
    size_t posSize = numParticles * 3 * sizeof(float);
    size_t velSize = numParticles * 3 * sizeof(float);
    size_t forceSize = numParticles * 3 * sizeof(float);
    size_t scalarSize = numParticles * sizeof(float);
    
    cudaError_t error;
    
    error = cudaMalloc(&d_positions, posSize);
    if (error != cudaSuccess) {
        throw std::runtime_error("Failed to allocate GPU memory for positions: " + std::string(cudaGetErrorString(error)));
    }
    
    error = cudaMalloc(&d_velocities, velSize);
    if (error != cudaSuccess) {
        throw std::runtime_error("Failed to allocate GPU memory for velocities: " + std::string(cudaGetErrorString(error)));
    }
    
    error = cudaMalloc(&d_forces, forceSize);
    if (error != cudaSuccess) {
        throw std::runtime_error("Failed to allocate GPU memory for forces: " + std::string(cudaGetErrorString(error)));
    }
    
    error = cudaMalloc(&d_densities, scalarSize);
    if (error != cudaSuccess) {
        throw std::runtime_error("Failed to allocate GPU memory for densities: " + std::string(cudaGetErrorString(error)));
    }
    
    error = cudaMalloc(&d_pressures, scalarSize);
    if (error != cudaSuccess) {
        throw std::runtime_error("Failed to allocate GPU memory for pressures: " + std::string(cudaGetErrorString(error)));
    }
    
    error = cudaMemcpy(d_positions, hostParticlePositions.data(), posSize, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        throw std::runtime_error("Failed to copy initial positions to GPU: " + std::string(cudaGetErrorString(error)));
    }
    
    cudaMemset(d_velocities, 0, velSize);
    cudaMemset(d_forces, 0, forceSize);
    cudaMemset(d_densities, 0, scalarSize);
    cudaMemset(d_pressures, 0, scalarSize);
    
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error("CUDA buffer initialization failed: " + std::string(cudaGetErrorString(error)));
    }
    
    Logger::info("CUDA buffers initialized successfully");
    Logger::info("GPU memory allocated: " + std::to_string((posSize + velSize + forceSize + 2*scalarSize) / (1024*1024)) + " MB");
}

void SphFluidModel::update(float deltaTime) {
    float remainingTime = deltaTime;
    while (remainingTime > 0.0f) {
        float dt = std::min(remainingTime, timeStep);
        
        computeDensityPressure();
        computeForces();
        integrate(dt);
        handleBoundaryConditions();
        
        remainingTime -= dt;
    }
    
    copyDataFromGPU();
}

void SphFluidModel::computeDensityPressure() {
    launch_density_pressure_kernel(
        d_positions,
        d_densities,
        d_pressures,
        numParticles
    );
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        Logger::error("CUDA density/pressure kernel error: " + std::string(cudaGetErrorString(error)));
    }
}

void SphFluidModel::computeForces() {
    launch_forces_kernel(
        d_positions,
        d_velocities,
        d_densities,
        d_pressures,
        d_forces,
        numParticles
    );
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        Logger::error("CUDA forces kernel error: " + std::string(cudaGetErrorString(error)));
    }
}

void SphFluidModel::integrate(float dt) {
    launch_integrate_kernel(
        d_positions,
        d_velocities,
        d_forces,
        dt,
        numParticles
    );
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        Logger::error("CUDA integration kernel error: " + std::string(cudaGetErrorString(error)));
    }
}

void SphFluidModel::handleBoundaryConditions() {
    float restitution = 0.5f;
    launch_boundary_kernel(
        d_positions,
        d_velocities,
        restitution,
        numParticles
    );
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        Logger::error("CUDA boundary kernel error: " + std::string(cudaGetErrorString(error)));
    }
}

void SphFluidModel::copyDataFromGPU() const {
    cudaError_t error = cudaMemcpy(
        const_cast<glm::vec3*>(hostParticlePositions.data()), 
        d_positions, 
        numParticles * 3 * sizeof(float), 
        cudaMemcpyDeviceToHost
    );
    
    if (error != cudaSuccess) {
        Logger::error("Failed to copy positions from GPU: " + std::string(cudaGetErrorString(error)));
        return;
    }
    
    updateRenderData();
}

void SphFluidModel::updateRenderData() const {
    for (int i = 0; i < numParticles; ++i) {
        // Pos
        hostParticleRenderData[i * 6 + 0] = hostParticlePositions[i].x;
        hostParticleRenderData[i * 6 + 1] = hostParticlePositions[i].y;
        hostParticleRenderData[i * 6 + 2] = hostParticlePositions[i].z;
        
        // Color based on height
        float height = (hostParticlePositions[i].y - boxMin.y) / (boxMax.y - boxMin.y);
        height = std::clamp(height, 0.0f, 1.0f);
        
        hostParticleRenderData[i * 6 + 3] = 0.1f + height * 0.3f; // R
        hostParticleRenderData[i * 6 + 4] = 0.3f + height * 0.5f; // G
        hostParticleRenderData[i * 6 + 5] = 0.8f + height * 0.2f; // B
    }
}

const std::vector<glm::vec3>& SphFluidModel::getParticlePositions() const {
    return hostParticlePositions;
}

const std::vector<float>& SphFluidModel::getParticleRenderData() const {
    return hostParticleRenderData;
}

void SphFluidModel::shutdown() {    
    freeCudaBuffers();
    hostParticlePositions.clear();
    hostParticleRenderData.clear();
    
    Logger::info("SPH Fluid Model shutdown complete");
}

void SphFluidModel::freeCudaBuffers() {
    if (d_positions) { 
        cudaFree(d_positions); 
        d_positions = nullptr; 
    }

    if (d_velocities) { 
        cudaFree(d_velocities); 
        d_velocities = nullptr; 
    }
    
    if (d_forces) { 
        cudaFree(d_forces); 
        d_forces = nullptr; 
    }
    
    if (d_densities) { 
        cudaFree(d_densities); 
        d_densities = nullptr; 
    }
    
    if (d_pressures) { 
        cudaFree(d_pressures); 
        d_pressures = nullptr; 
    }
    
    cudaDeviceSynchronize();
}

bool SphFluidModel::validateParameters() const {
    if (numParticles <= 0) {
        Logger::error("Invalid number of particles: " + std::to_string(numParticles));
        return false;
    }
    
    if (particleMass <= 0.0f || restDensity <= 0.0f || smoothingRadius <= 0.0f) {
        Logger::error("Invalid physical parameters");
        return false;
    }
    
    glm::vec3 boxSize = boxMax - boxMin;
    if (boxSize.x <= 0.0f || boxSize.y <= 0.0f || boxSize.z <= 0.0f) {
        Logger::error("Invalid simulation box dimensions");
        return false;
    }
    
    return true;
}

void SphFluidModel::printSimulationInfo() const {
    Logger::info("SPH Simulation Parameters:");
    Logger::info("Particles: " + std::to_string(numParticles));
    Logger::info("Particle mass: " + std::to_string(particleMass) + " kg");
    Logger::info("Rest density: " + std::to_string(restDensity) + " kg/m³");
    Logger::info("Gas constant: " + std::to_string(gasConstant));
    Logger::info("Viscosity: " + std::to_string(viscosity));
    Logger::info("Smoothing radius: " + std::to_string(smoothingRadius) + " m");
    Logger::info("Time step: " + std::to_string(timeStep) + " s");
    Logger::info("Particle spacing: " + std::to_string(spacing) + " m");
    
    glm::vec3 boxSize = boxMax - boxMin;
    Logger::info("Simulation box: " + 
                std::to_string(boxSize.x) + " x " + 
                std::to_string(boxSize.y) + " x " + 
                std::to_string(boxSize.z) + " m");
    Logger::info("CUDA Kernels: ENABLED");
}
