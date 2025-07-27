#include "FluidSimulator.h"
#include "utils/Logger.h"

// TODO: Incluir archivos de cabecera de CUDA necesarios aquí (ej. si usas funciones de tiempo de ejecución CUDA)
#include <cuda_runtime.h> // Para cudaGetDeviceProperties, cudaSetDevice, etc.

// Incluir archivos de cabecera de las extensiones de PhysX necesarias para CUDA
#include <extensions/PxExtensionsAPI.h> // Para PxCreateCudaContextManager

// Usar el namespace de PhysX para acortar
using namespace physx;

// Constructor
FluidSimulator::FluidSimulator()
    : gErrorCallback(nullptr), gAllocator(nullptr), gFoundation(nullptr), gPhysics(nullptr),
      gPvd(nullptr), gDispatcher(nullptr), gScene(nullptr), gMaterial(nullptr),
      gCudaContextManager(nullptr), activeFluidModel(nullptr),
      accumulator(0.0f), stepSize(1.0f / 60.0f)
{
}

// Destructor
FluidSimulator::~FluidSimulator() {
    shutdown();
}

// Inicialización del simulador de fluidos
bool FluidSimulator::initialize() {
    Logger::info("Initializing PhysX 5.6 fluid simulator with CUDA support and multiple fluid types...");

    gErrorCallback = PxDefaultErrorCallback();
    gAllocator = PxDefaultAllocator();

    gFoundation = PxCreateFoundation(PX_PHYSICS_VERSION, *gAllocator, *gErrorCallback);
    if (!gFoundation) {
        Logger::error("PxCreateFoundation failed!");
        return false;
    }

    gPvd = PxCreatePvd(*gFoundation);
    PxPvdTransport* transport = PxDefaultPvdSocketTransportCreate("127.0.0.1", 5425, 10);
    if (transport) {
        gPvd->connect(*transport, PxPvdInstrumentationFlag::eALL);
        Logger::info("Connected to PhysX Visual Debugger (PVD).");
    } else {
        Logger::warn("Failed to connect to PhysX Visual Debugger (PVD). Check if PVD is running.");
    }

    // --- Inicialización de CUDA ---
    // PxCreateCudaContextManager requiere el PxExtensionsAPI y las librerías de CUDA
    gCudaContextManager = PxCreateCudaContextManager(*gFoundation);
    if (!gCudaContextManager) {
        Logger::error("Failed to create PxCudaContextManager. Is CUDA Toolkit installed and configured?");
        // Si CUDA es crucial, puedes decidir si retornar false aquí o continuar sin GPU.
        return false;
    }
    if (!gCudaContextManager->contextIsValid()) {
        Logger::error("CUDA context is not valid!");
        // Esto puede pasar si no hay GPU compatible o drivers incorrectos.
        return false;
    }
    Logger::info("CUDA Context Manager created successfully.");

    // TODO: Opcional: Configurar la GPU si tienes varias (cudaSetDevice)
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount > 0) {
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, 0); // Usar el dispositivo 0
        Logger::info("CUDA Device 0: " + std::string(devProp.name) + " (Compute Capability " + std::to_string(devProp.major) + "." + std::to_string(devProp.minor) + ")");
        // cudaSetDevice(0); // Si quieres forzar el uso de un dispositivo específico
    }


    gPhysics = PxCreatePhysics(PX_PHYSICS_VERSION, *gFoundation, PxTolerancesScale(), true, gPvd);
    if (!gPhysics) {
        Logger::error("PxCreatePhysics failed!");
        return false;
    }

    gDispatcher = PxDefaultCpuDispatcherCreate(4); // 4 hilos CPU para PhysX
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

    // TODO: Ahora, en lugar de inicializar directamente las partículas,
    // registraremos y activaremos un modelo de fluido.
    // addFluidModel("SPH Water", new SphFluidModel()); // Ejemplo: Necesitarás implementar SphFluidModel
    // setActiveFluidModel("SPH Water");

    Logger::info("PhysX 5.6 fluid simulator with CUDA initialized. Ready to load fluid models.");
    return true;
}

// Simulación de un paso de tiempo
void FluidSimulator::simulate(float deltaTime) {
    accumulator += deltaTime;
    while (accumulator >= stepSize) {
        // Primero, actualizar la lógica de tu modelo de fluido (SPH, etc.)
        if (activeFluidModel) {
            activeFluidModel->update(stepSize); // El modelo de fluido maneja su propia simulación (CUDA kernels)
        }

        // Luego, simular la escena de PhysX para colisiones de partículas/rígidos
        gScene->simulate(stepSize);
        gScene->fetchResults(true);

        accumulator -= stepSize;

        // Opcional: Si el modelo de fluido necesita datos de PhysX (ej. contacto), los leerá aquí.
    }
}

// Apagado del simulador de fluidos
void FluidSimulator::shutdown() {
    Logger::info("Shutting down PhysX 5.6 fluid simulator...");

    // Apagar y liberar todos los modelos de fluido
    for (auto const& [name, model] : fluidModels) {
        Logger::info("Shutting down fluid model: " + name);
        model->shutdown();
        delete model;
    }
    fluidModels.clear();
    activeFluidModel = nullptr;

    // Liberar recursos de PhysX y CUDA en el orden correcto
    // (Pase gFluidParticleRigidSet si se maneja aquí o en el modelo de fluido)

    if (gScene) { gScene->release(); gScene = nullptr; }
    if (gDispatcher) { gDispatcher->release(); gDispatcher = nullptr; }
    if (gMaterial) { gMaterial->release(); gMaterial = nullptr; }

    // Liberar PxCudaContextManager antes de PxPhysics y PxFoundation
    if (gCudaContextManager) {
        gCudaContextManager->release();
        gCudaContextManager = nullptr;
    }

    if (gPvd) {
        PxPvdTransport* transport = gPvd->getTransport();
        gPvd->release(); gPvd = nullptr;
        if (transport) { transport->release(); }
    }
    if (gPhysics) { gPhysics->release(); gPhysics = nullptr; }
    if (gFoundation) { gFoundation->release(); gFoundation = nullptr; }

    Logger::info("PhysX 5.6 fluid simulator shutdown complete.");
}

// Métodos para gestionar modelos de fluido
bool FluidSimulator::addFluidModel(const std::string& name, IFluidModel* model) {
    if (fluidModels.count(name) > 0) {
        Logger::error("Fluid model with name '" + name + "' already exists.");
        delete model; // Evitar fuga de memoria si ya existe
        return false;
    }
    fluidModels[name] = model;
    model->name = name; // Asignar el nombre al modelo
    Logger::info("Added fluid model: " + name);
    return true;
}

bool FluidSimulator::setActiveFluidModel(const std::string& name) {
    auto it = fluidModels.find(name);
    if (it != fluidModels.end()) {
        activeFluidModel = it->second;
        Logger::info("Active fluid model set to: " + name);
        return true;
    }
    Logger::error("Fluid model '" + name + "' not found.");
    return false;
}

const std::vector<glm::vec3>& FluidSimulator::getActiveFluidParticlePositions() const {
    if (activeFluidModel) {
        return activeFluidModel->getParticlePositions();
    }
    static const std::vector<glm::vec3> emptyVec; // Retorna un vector vacío si no hay modelo activo
    return emptyVec;
}


// --- Métodos Privados ---

void FluidSimulator::configureScene() {
    PxSceneDesc sceneDesc(gPhysics->getTolerancesScale());
    sceneDesc.gravity = PxVec3(0.0f, -9.81f, 0.0f);
    sceneDesc.cpuDispatcher = gDispatcher;

    // TODO: Configuración de la GPU para la escena si se usa PxParticleRigidSet en la escena
    // sceneDesc.cudaContextManager = gCudaContextManager; // Necesario si quieres PhysX en GPU

    // Opcional: Un filtro de simulación por defecto
    sceneDesc.filterShader = PxDefaultSimulationFilterShader;

    gScene = gPhysics->createScene(sceneDesc);
    if (!gScene) {
        Logger::error("Failed to create PxScene!");
    }

    if (gPvd) {
        gScene->getScenePvdClient()->connect(*gPvd->getTransport(), PxPvdInstrumentationFlag::eALL);
    }
}

void FluidSimulator::createFluidContainer() {
    // Ejemplo: Crear un plano como suelo
    PxRigidStatic* groundPlane = PxCreateStatic(
        *gPhysics,
        PxTransform(PxVec3(0.0f, 0.0f, 0.0f), PxQuat(PxIdentity)),
        PxPlaneGeometry(),
        *gMaterial
    );
    if (groundPlane) {
        gScene->addActor(*groundPlane);
        Logger::info("Ground plane added to PhysX scene.");
    } else {
        Logger::error("Failed to create ground plane.");
    }

    // TODO: Crear paredes como PxRigidStatic con PxBoxGeometry para un contenedor cerrado.
    Logger::warn("Fluid container walls not yet fully implemented (TODO).");
}

// Este método ahora se encarga de crear e inicializar PxParticleRigidSet
// y preparar el entorno para los modelos de fluido específicos.
void FluidSimulator::initializeFluidParticles() {
    // La inicialización de partículas ahora se delega a los IFluidModel concretos.
    // Aquí podrías añadir un PxParticleRigidSet global si todos los modelos lo comparten.
    // O cada IFluidModel podría gestionar el suyo si las configuraciones son muy diferentes.
    // Por simplicidad, podríamos tener un único PxParticleRigidSet gestionado aquí
    // si solo es para las colisiones entre partículas de fluidos y rígidos.

    // PxU32 maxParticles = 100000; // Capacidad máxima global de partículas
    // gFluidParticleRigidSet = gPhysics->createParticleRigidSet(maxParticles);
    // if (!gFluidParticleRigidSet) {
    //     Logger::error("Failed to create PxParticleRigidSet for fluid collision.");
    //     return;
    // }
    // gScene->addActor(*gFluidParticleRigidSet);
    // Logger::info("Global PxParticleRigidSet created for fluid collision detection.");
    Logger::warn("Particle system setup for fluid models is now handled by individual IFluidModel implementations.");
}

// Este método es obsoleto en esta estructura; la gestión de render data es de IFluidModel
// void FluidSimulator::updateFluidParticleRenderData() {} // REMOVED as it's now in IFluidModel