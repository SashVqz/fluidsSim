// src/physics/SphFluidModel.h - Versión actualizada con kernels CUDA
#pragma once

#include "FluidSimulator.h"
#include <cuda_runtime.h>
#include <vector>
#include <glm/glm.hpp>

class SphFluidModel : public IFluidModel {
public:
    SphFluidModel(int numParticles, const glm::vec3& boxMin, const glm::vec3& boxMax);
    ~SphFluidModel() override;

    bool initialize(physx::PxPhysics* physics = nullptr, physx::PxScene* scene = nullptr, physx::PxCudaContextManager* cudaContextManager = nullptr) override;
    void update(float deltaTime) override;
    const std::vector<glm::vec3>& getParticlePositions() const override;
    void shutdown() override;

    const std::vector<float>& getParticleRenderData() const;
    int getNumParticles() const { return numParticles; }

private:
    int numParticles;
    glm::vec3 boxMin, boxMax;

    // host buffers (CPU)
    mutable std::vector<glm::vec3> hostParticlePositions;
    mutable std::vector<float> hostParticleRenderData; // pos + colors interleaved

    // device Buffers (GPU)
    float* d_positions;       // 3 * numParticles floats (x,y,z for each particle)
    float* d_velocities;      // 3 * numParticles floats (vx,vy,vz for each particle)
    float* d_forces;          // 3 * numParticles floats (fx,fy,fz for each particle)
    float* d_densities;       // numParticles floats
    float* d_pressures;       // numParticles floats
    
    // SPH fluid parameters
    float particleMass;       // Mass for each particle
    float restDensity;        // Densidad de referencia del fluido
    float gasConstant;        // Constante de gas para ecuación de estado
    float viscosity;          // Coeficiente de viscosidad
    float smoothingRadius;    // Radio de influencia para kernels SPH
    float timeStep;           // Paso de tiempo de la simulación
    float dampingFactor;      // Factor de amortiguación para estabilidad
    float spacing;            // Espaciado inicial entre partículas
    
    physx::PxPhysics* pxPhysics;
    physx::PxScene* pxScene;
    physx::PxCudaContextManager* pxCudaContextManager;
    
    void initializeParticles();
    void initializeCudaBuffers();
    void freeCudaBuffers();
    void setupCudaConstants();
    
    void computeDensityPressure();
    void computeForces();
    void integrate(float dt);
    void handleBoundaryConditions();
    
    void copyDataFromGPU() const;
    void updateRenderData() const;
    
    bool validateParameters() const;
    void printSimulationInfo() const;
};