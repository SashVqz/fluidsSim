// src/physics/sph_kernels.h
#pragma once

#include <cuda_runtime.h>

// host
extern "C" {
    void setup_sph_constants(
        float smoothingRadius,
        float particleMass,
        float restDensity,
        float gasConstant,
        float viscosity,
        float dampingFactor,
        float3 gravity,
        float3 boxMin,
        float3 boxMax
    );
    
    void launch_density_pressure_kernel(
        float* d_positions,    // Device pointer - posiciones [numParticles * 3]
        float* d_densities,    // Device pointer - densidades [numParticles]
        float* d_pressures,    // Device pointer - presiones [numParticles]
        int numParticles
    );
    
    void launch_forces_kernel(
        float* d_positions,    // Device pointer - posiciones [numParticles * 3]
        float* d_velocities,   // Device pointer - velocidades [numParticles * 3]
        float* d_densities,    // Device pointer - densidades [numParticles]
        float* d_pressures,    // Device pointer - presiones [numParticles]
        float* d_forces,       // Device pointer - fuerzas [numParticles * 3]
        int numParticles
    );
    
    void launch_integrate_kernel(
        float* d_positions,    // Device pointer - posiciones [numParticles * 3]
        float* d_velocities,   // Device pointer - velocidades [numParticles * 3]
        float* d_forces,       // Device pointer - fuerzas [numParticles * 3]
        float dt,              // Paso de tiempo
        int numParticles
    );
    
    void launch_boundary_kernel(
        float* d_positions,    // Device pointer - posiciones [numParticles * 3]
        float* d_velocities,   // Device pointer - velocidades [numParticles * 3]
        float restitution,     // Coeficiente de restitución
        int numParticles
    );
}