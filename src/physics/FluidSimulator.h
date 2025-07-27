#pragma once

#include <PxPhysicsAPI.h>
#include <foundation/PxAssert.h>
#include <pvd/PxPvd.h>

#include <glm/glm.hpp>
#include <vector>
#include <string>
#include <map> // Para gestionar múltiples modelos de fluido

// Forward declarations de PhysX
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
    class PxCudaContextManager; // Para la gestión del contexto CUDA
}

// --- Interfaz para Modelos de Fluidos ---
// Una clase base abstracta para diferentes tipos de simulación de fluidos.
// Cada implementación concreta (ej. SPH, MPM) contendrá su lógica y,
// probablemente, sus propios buffers de CUDA.
class IFluidModel {
public:
    virtual ~IFluidModel() = default;

    // TODO: Inicializar el modelo de fluido (buffers CUDA, parámetros, etc.)
    // Toma la instancia de PxPhysics y el PxCudaContextManager para la interacción.
    virtual bool initialize(physx::PxPhysics* physics, physx::PxScene* scene, physx::PxCudaContextManager* cudaContextManager) = 0;

    // TODO: Actualizar la simulación del fluido
    virtual void update(float deltaTime) = 0;

    // TODO: Obtener las posiciones de las partículas para el renderizado
    // Puede necesitar copiar de la GPU a la CPU, o se renderizará directamente desde un buffer GPU.
    virtual const std::vector<glm::vec3>& getParticlePositions() const = 0;

    // TODO: Obtener cualquier otra propiedad del fluido para depuración/renderizado (densidad, color, etc.)
    // virtual const std::vector<float>& getParticleDensities() const = 0;

    // TODO: Liberar recursos del modelo de fluido
    virtual void shutdown() = 0;

    // Propiedades comunes que podrían ser gestionadas por la interfaz
    std::string name; // Nombre del tipo de fluido (e.g., "Water", "Oil")
};

// --- Clase del Simulador Principal (Orquestador) ---
class FluidSimulator {
public:
    FluidSimulator();
    ~FluidSimulator();

    // Inicializa PhysX y, opcionalmente, CUDA
    bool initialize();
    void simulate(float deltaTime);
    void shutdown();

    // TODO: Métodos para gestionar modelos de fluido
    // Añadir un nuevo tipo de modelo de fluido
    bool addFluidModel(const std::string& name, IFluidModel* model);
    // Cambiar el modelo de fluido activo
    bool setActiveFluidModel(const std::string& name);
    // Obtener las posiciones de las partículas del modelo de fluido activo para el renderizado
    const std::vector<glm::vec3>& getActiveFluidParticlePositions() const;

private:
    physx::PxDefaultErrorCallback* gErrorCallback;
    physx::PxDefaultAllocator* gAllocator;
    physx::PxFoundation* gFoundation;
    physx::PxPhysics* gPhysics;
    physx::PxPvd* gPvd;
    physx::PxDefaultCpuDispatcher* gDispatcher;
    physx::PxScene* gScene;
    physx::PxMaterial* gMaterial;

    // --- Gestión de CUDA ---
    // El Context Manager es esencial para usar PhysX en GPU y para tus propios kernels CUDA.
    physx::PxCudaContextManager* gCudaContextManager;

    // --- Gestión de Modelos de Fluidos ---
    std::map<std::string, IFluidModel*> fluidModels; // Mapa de todos los modelos de fluido disponibles
    IFluidModel* activeFluidModel;                   // El modelo de fluido que se está simulando actualmente

    // TODO: Contenedor para el fluido (límites de la simulación)
    // physx::PxRigidStatic* boundaryActors[6];

    float accumulator;
    float stepSize;

    // Métodos privados
    void configureScene();
    void createFluidContainer();
};


// --- Ejemplo de Implementación Concreta de un Modelo de Fluido (SPH) ---
// Puedes definir esto en un archivo separado, por ejemplo, SphFluidModel.h/.cpp
// para mantener el FluidSimulator.h limpio.
/*
#include <cuda_runtime.h> // Para funciones de CUDA

class SphFluidModel : public IFluidModel {
public:
    SphFluidModel();
    ~SphFluidModel() override; // Usar override para indicar que es una función virtual sobrescrita

    bool initialize(physx::PxPhysics* physics, physx::PxScene* scene, physx::PxCudaContextManager* cudaContextManager) override;
    void update(float deltaTime) override;
    const std::vector<glm::vec3>& getParticlePositions() const override;
    void shutdown() override;

private:
    // Punteros a datos de partículas en la GPU (buffers de CUDA)
    float* d_positions;
    float* d_velocities;
    // ... otros buffers para densidad, presión, fuerzas, etc.

    // Cantidad de partículas
    int numParticles;

    // Referencias a objetos de PhysX y CUDA para su uso interno
    physx::PxPhysics* pxPhysics;
    physx::PxScene* pxScene;
    physx::PxCudaContextManager* pxCudaContextManager;

    // Buffers para copiar datos de GPU a CPU para renderizado
    mutable std::vector<glm::vec3> hostParticlePositions;

    // TODO: Parámetros SPH (masa, radio de suavizado, densidad de referencia, viscosidad, etc.)
    float particleMass;
    float smoothingRadius;

    // TODO: Kernel CUDA para inicialización de partículas
    void initCudaParticles();
    // TODO: Kernel CUDA para calcular densidades y presiones
    void calculateDensitiesAndPressures();
    // TODO: Kernel CUDA para calcular fuerzas
    void calculateForces();
    // TODO: Kernel CUDA para integrar (actualizar posiciones y velocidades)
    void integrate();
    // TODO: Kernel CUDA para manejar colisiones con PhysX rígidos (si no lo hace PxParticleRigidSet automáticamente)
    // Esto podría implicar leer datos de colisión de PhysX y ajustar las velocidades de las partículas.
};
*/