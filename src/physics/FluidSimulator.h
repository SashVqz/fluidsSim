#pragma once

#include <PxPhysicsAPI.h>
#include <foundation/PxAssert.h>
#include <pvd/PxPvd.h>
#include <extensions/PxExtensionsAPI.h>

#include <glm/glm.hpp>
#include <vector>
#include <string>
#include <map>
#include <memory>

namespace physx {
    class PxFoundation;
    class PxPhysics;
    class PxPvd;
    class PxScene;
    class PxDefaultCpuDispatcher;
    class PxDefaultErrorCallback;
    class PxDefaultAllocator;
    class PxMaterial;
    class PxParticleRigidSet;
    class PxCudaContextManager;
}

class IFluidModel {
public:
    virtual ~IFluidModel() = default;
    virtual bool initialize(physx::PxPhysics* physics = nullptr, physx::PxScene* scene = nullptr, physx::PxCudaContextManager* cudaContextManager = nullptr) = 0;
    virtual void update(float deltaTime) = 0;
    virtual const std::vector<glm::vec3>& getParticlePositions() const = 0;
    virtual void shutdown() = 0;

    std::string name; // water right now
};

class FluidSimulator {
public:
    FluidSimulator();
    ~FluidSimulator();

    bool initialize();
    void simulate(float deltaTime);
    void shutdown();
    bool addFluidModel(const std::string& name, IFluidModel* model);
    bool setActiveFluidModel(const std::string& name);
    const std::vector<glm::vec3>& getActiveFluidParticlePositions() const;

private:
    std::unique_ptr<physx::PxDefaultErrorCallback> gErrorCallback;
    std::unique_ptr<physx::PxDefaultAllocator> gAllocator;
    physx::PxFoundation* gFoundation;
    physx::PxPhysics* gPhysics;
    physx::PxPvd* gPvd;
    physx::PxDefaultCpuDispatcher* gDispatcher;
    physx::PxScene* gScene;
    physx::PxMaterial* gMaterial;
    physx::PxCudaContextManager* gCudaContextManager;

    std::map<std::string, std::unique_ptr<IFluidModel>> fluidModels;
    IFluidModel* activeFluidModel;

    float accumulator;
    float stepSize;

    void configureScene();
    void createFluidContainer();
};