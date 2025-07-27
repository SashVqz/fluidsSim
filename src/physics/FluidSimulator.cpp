#include "FluidSimulator.h"
#include "utils/Logger.h"
#include <cuda_runtime.h> 
#include <extensions/PxExtensionsAPI.h>
#include <gpu/PxGpu.h>

using namespace physx;

FluidSimulator::FluidSimulator()
    : gFoundation(nullptr), gPhysics(nullptr),
      gPvd(nullptr), gDispatcher(nullptr), gScene(nullptr), gMaterial(nullptr),
      gCudaContextManager(nullptr), activeFluidModel(nullptr),
      accumulator(0.0f), stepSize(1.0f / 60.0f)
{
    gErrorCallback = std::make_unique<PxDefaultErrorCallback>();
    gAllocator = std::make_unique<PxDefaultAllocator>();
}

FluidSimulator::~FluidSimulator() {
    shutdown();
}

bool FluidSimulator::initialize() {
    Logger::info("Initializing PhysX 5.6 fluid simulator with CUDA support and multiple fluid types...");

    gFoundation = PxCreateFoundation(PX_PHYSICS_VERSION, *gAllocator, *gErrorCallback);
    if (!gFoundation) {
        Logger::error("PxCreateFoundation failed!");
        return false;
    }

    gPvd = PxCreatePvd(*gFoundation);
    if (gPvd) {
        PxPvdTransport* transport = PxDefaultPvdSocketTransportCreate("127.0.0.1", 5425, 10);
        if (transport) {
            gPvd->connect(*transport, PxPvdInstrumentationFlag::eALL);
            Logger::info("Connected to PhysX Visual Debugger (PVD).");
        } else Logger::warn("Failed to connect to PhysX Visual Debugger (PVD). Check if PVD is running.");
    }

    PxCudaContextManagerDesc cudaContextManagerDesc;
    gCudaContextManager = PxCreateCudaContextManager(*gFoundation, cudaContextManagerDesc);
    if (!gCudaContextManager) {
        Logger::error("Failed to create PxCudaContextManager. Is CUDA Toolkit installed and configured?");
        return false;
    }

    if (!gCudaContextManager->contextIsValid()) {
        Logger::error("CUDA context is not valid!");
        return false;
    }
    
    Logger::info("CUDA Context Manager created successfully.");

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount > 0) {
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, 0);

        Logger::info("CUDA Device 0: " + std::string(devProp.name) + " (Compute Capability " + std::to_string(devProp.major) + "." + std::to_string(devProp.minor) + ")");
        
        // cudaSetDevice(0);
    }

    gPhysics = PxCreatePhysics(PX_PHYSICS_VERSION, *gFoundation, PxTolerancesScale(), true, gPvd);
    if (!gPhysics) {
        Logger::error("PxCreatePhysics failed!");
        return false;
    }

    gDispatcher = PxDefaultCpuDispatcherCreate(4); // 4 CPU threads for PhysX 
    if (!gDispatcher) {
        Logger::error("PxDefaultCpuDispatcherCreate failed!");
        return false;
    }

    configureScene();

    gMaterial = gPhysics->createMaterial(0.5f, 0.5f, 0.6f);
    if (!gMaterial) {
        Logger::error("Failed to create PxMaterial!");
        return false;
    }

    createFluidContainer();

    Logger::info("PhysX 5.6 fluid simulator with CUDA initialized. Ready to load fluid models.");
    return true;
}

void FluidSimulator::simulate(float deltaTime) {
    accumulator += deltaTime;
    while (accumulator >= stepSize) {
        if (activeFluidModel) {
            activeFluidModel->update(stepSize);
        }

        gScene->simulate(stepSize);
        gScene->fetchResults(true);

        accumulator -= stepSize;
    }
}

void FluidSimulator::shutdown() {
    Logger::info("Shutting down fluid simulator");

    for (auto& [name, model] : fluidModels) {
        Logger::info("Shutting down fluid model: " + name);
        model->shutdown();
    }

    fluidModels.clear();
    activeFluidModel = nullptr;

    
    if (gScene) { 
        gScene->release(); 
        gScene = nullptr; 
    }
    
    if (gDispatcher) { 
        gDispatcher->release(); 
        gDispatcher = nullptr; 
    }
    
    if (gMaterial) { 
        gMaterial->release(); 
        gMaterial = nullptr; 
    }

    if (gCudaContextManager) {
        gCudaContextManager->release();
        gCudaContextManager = nullptr;
    }

    if (gPvd) {
        PxPvdTransport* transport = gPvd->getTransport();
        gPvd->release(); 
        gPvd = nullptr;
        if (transport) transport->release(); 
    }

    if (gPhysics) { 
        gPhysics->release(); 
        gPhysics = nullptr; 
    }
    
    if (gFoundation) { 
        gFoundation->release(); 
        gFoundation = nullptr; 
    }

    Logger::info("fluid simulator shutdown complete.");
}

bool FluidSimulator::addFluidModel(const std::string& name, IFluidModel* model) {
    if (fluidModels.count(name) > 0) {
        Logger::error("Fluid model with name '" + name + "' already exists.");
        delete model;
        return false;
    }
    
    fluidModels[name] = std::unique_ptr<IFluidModel>(model);
    model->name = name;
    Logger::info("Added fluid model: " + name);
    return true;
}

bool FluidSimulator::setActiveFluidModel(const std::string& name) {
    auto it = fluidModels.find(name);
    if (it != fluidModels.end()) {
        activeFluidModel = it->second.get(); 
        Logger::info("Active fluid model set to: " + name);
        return true;
    }
    Logger::error("Fluid model '" + name + "' not found.");
    return false;
}

const std::vector<glm::vec3>& FluidSimulator::getActiveFluidParticlePositions() const {
    if (activeFluidModel) return activeFluidModel->getParticlePositions();
    static const std::vector<glm::vec3> emptyVec;
    return emptyVec;
}

void FluidSimulator::configureScene() {
    PxSceneDesc sceneDesc(gPhysics->getTolerancesScale());
    sceneDesc.gravity = PxVec3(0.0f, -9.81f, 0.0f);
    sceneDesc.cpuDispatcher = gDispatcher;
    sceneDesc.cudaContextManager = gCudaContextManager;
    sceneDesc.filterShader = PxDefaultSimulationFilterShader;

    gScene = gPhysics->createScene(sceneDesc);
    if (!gScene) {
        Logger::error("Failed to create PxScene!");
        return;
    }

    if (gPvd) {
        PxPvdSceneClient* pvdClient = gScene->getScenePvdClient();
        if (pvdClient) {
            pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_CONSTRAINTS, true);
            pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_CONTACTS, true);
            pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_SCENEQUERIES, true);
        }
    }
}

void FluidSimulator::createFluidContainer() {
    PxRigidStatic* groundPlane = PxCreatePlane(*gPhysics, PxPlane(0, 1, 0, 0), *gMaterial);
    if (groundPlane) {
        gScene->addActor(*groundPlane);
        Logger::info("Ground plane added to PhysX scene.");
    } else {
        Logger::error("Failed to create ground plane.");
    }

    // TODO: Crear paredes como PxRigidStatic con PxBoxGeometry para un contenedor cerrado.
    Logger::warn("Fluid container walls not yet fully implemented (TODO).");
}