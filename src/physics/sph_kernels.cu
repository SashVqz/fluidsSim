#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

__constant__ float c_smoothingRadius;
__constant__ float c_smoothingRadius2;
__constant__ float c_particleMass;
__constant__ float c_restDensity;
__constant__ float c_gasConstant;
__constant__ float c_viscosity;
__constant__ float c_dampingFactor;
__constant__ float3 c_gravity;
__constant__ float3 c_boxMin;
__constant__ float3 c_boxMax;

__device__ float length_squared(float3 v) {
    return v.x * v.x + v.y * v.y + v.z * v.z;
}

__device__ float3 normalize_safe(float3 v) {
    float len = sqrtf(length_squared(v));
    if (len > 1e-6f) return make_float3(v.x / len, v.y / len, v.z / len);
    return make_float3(0.0f, 0.0f, 0.0f);
}

__device__ float poly6_kernel(float r_squared, float h) {
    if (r_squared >= h * h) return 0.0f;
    
    float h2 = h * h;
    float h9 = h2 * h2 * h2 * h2 * h;
    float diff = h2 - r_squared;
    return (315.0f / (64.0f * M_PI * h9)) * diff * diff * diff;
}

__device__ float3 spiky_gradient(float3 r_vec, float r, float h) {
    if (r >= h || r < 1e-6f) return make_float3(0.0f, 0.0f, 0.0f);
    
    float h6 = h * h * h * h * h * h;
    float diff = h - r;
    float factor = -45.0f / (M_PI * h6) * diff * diff / r;
    
    return make_float3(
        factor * r_vec.x,
        factor * r_vec.y,
        factor * r_vec.z
    );
}

__device__ float viscosity_laplacian(float r, float h) {
    if (r >= h) return 0.0f;
    
    float h6 = h * h * h * h * h * h;
    return (45.0f / (M_PI * h6)) * (h - r);
}

__global__ void compute_density_pressure_kernel(
    float* positions,      // [numParticles * 3]
    float* densities,      // [numParticles]
    float* pressures,      // [numParticles]
    int numParticles
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;
    
    float3 pos_i = make_float3(
        positions[i * 3 + 0],
        positions[i * 3 + 1],
        positions[i * 3 + 2]
    );
    
    float density = 0.0f;
    
    for (int j = 0; j < numParticles; ++j) {
        float3 pos_j = make_float3(
            positions[j * 3 + 0],
            positions[j * 3 + 1],
            positions[j * 3 + 2]
        );
        
        float3 r_vec = make_float3(
            pos_i.x - pos_j.x,
            pos_i.y - pos_j.y,
            pos_i.z - pos_j.z
        );
        
        float r_squared = length_squared(r_vec);
        
        if (r_squared < c_smoothingRadius2) {
            density += c_particleMass * poly6_kernel(r_squared, c_smoothingRadius);
        }
    }
    
    densities[i] = density;
    
    pressures[i] = c_gasConstant * (density - c_restDensity);
}

__global__ void compute_forces_kernel(
    float* positions,      // [numParticles * 3]
    float* velocities,     // [numParticles * 3]
    float* densities,      // [numParticles]
    float* pressures,      // [numParticles]
    float* forces,         // [numParticles * 3] - output
    int numParticles
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;
    
    float3 pos_i = make_float3(
        positions[i * 3 + 0],
        positions[i * 3 + 1],
        positions[i * 3 + 2]
    );
    
    float3 vel_i = make_float3(
        velocities[i * 3 + 0],
        velocities[i * 3 + 1],
        velocities[i * 3 + 2]
    );
    
    float density_i = densities[i];
    float pressure_i = pressures[i];
    
    float3 force_pressure = make_float3(0.0f, 0.0f, 0.0f);
    float3 force_viscosity = make_float3(0.0f, 0.0f, 0.0f);
    
    for (int j = 0; j < numParticles; ++j) {
        if (i == j) continue;
        
        float3 pos_j = make_float3(
            positions[j * 3 + 0],
            positions[j * 3 + 1],
            positions[j * 3 + 2]
        );
        
        float3 vel_j = make_float3(
            velocities[j * 3 + 0],
            velocities[j * 3 + 1],
            velocities[j * 3 + 2]
        );
        
        float3 r_vec = make_float3(
            pos_i.x - pos_j.x,
            pos_i.y - pos_j.y,
            pos_i.z - pos_j.z
        );
        
        float r = sqrtf(length_squared(r_vec));
        
        if (r < c_smoothingRadius && r > 1e-6f) {
            float density_j = densities[j];
            float pressure_j = pressures[j];
            
            // pressure force
            float3 pressure_grad = spiky_gradient(r_vec, r, c_smoothingRadius);
            float pressure_factor = c_particleMass * (pressure_i + pressure_j) / (2.0f * density_j);
            
            force_pressure.x -= pressure_factor * pressure_grad.x;
            force_pressure.y -= pressure_factor * pressure_grad.y;
            force_pressure.z -= pressure_factor * pressure_grad.z;
            
            // viscosity force
            float viscosity_lap = viscosity_laplacian(r, c_smoothingRadius);
            float viscosity_factor = c_viscosity * c_particleMass / density_j * viscosity_lap;
            
            force_viscosity.x += viscosity_factor * (vel_j.x - vel_i.x);
            force_viscosity.y += viscosity_factor * (vel_j.y - vel_i.y);
            force_viscosity.z += viscosity_factor * (vel_j.z - vel_i.z);
        }
    }
    
    // gravity force (fg = m * g)
    float3 force_gravity = make_float3(
        c_gravity.x * c_particleMass,
        c_gravity.y * c_particleMass,
        c_gravity.z * c_particleMass
    );
    
    // force total
    forces[i * 3 + 0] = force_pressure.x + force_viscosity.x + force_gravity.x;
    forces[i * 3 + 1] = force_pressure.y + force_viscosity.y + force_gravity.y;
    forces[i * 3 + 2] = force_pressure.z + force_viscosity.z + force_gravity.z;
}

// temporal integration kernel
__global__ void integrate_kernel(
    float* positions,      // [numParticles * 3] - input/output
    float* velocities,     // [numParticles * 3] - input/output
    float* forces,         // [numParticles * 3] - input
    float dt,
    int numParticles
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;
    
    float3 pos = make_float3(
        positions[i * 3 + 0],
        positions[i * 3 + 1],
        positions[i * 3 + 2]
    );
    
    float3 vel = make_float3(
        velocities[i * 3 + 0],
        velocities[i * 3 + 1],
        velocities[i * 3 + 2]
    );
    
    float3 force = make_float3(
        forces[i * 3 + 0],
        forces[i * 3 + 1],
        forces[i * 3 + 2]
    );
    
    // newton's second law: a = F/m
    float3 accel = make_float3(
        force.x / c_particleMass,
        force.y / c_particleMass,
        force.z / c_particleMass
    );
    
    // euler integration: v = v + a*dt
    vel.x += accel.x * dt;
    vel.y += accel.y * dt;
    vel.z += accel.z * dt;
    
    // apply damping
    vel.x *= c_dampingFactor;
    vel.y *= c_dampingFactor;
    vel.z *= c_dampingFactor;

    // position integration: x = x + v*dt
    pos.x += vel.x * dt;
    pos.y += vel.y * dt;
    pos.z += vel.z * dt;
    
    // update attributes
    positions[i * 3 + 0] = pos.x;
    positions[i * 3 + 1] = pos.y;
    positions[i * 3 + 2] = pos.z;
    
    velocities[i * 3 + 0] = vel.x;
    velocities[i * 3 + 1] = vel.y;
    velocities[i * 3 + 2] = vel.z;
}

// boundary conditions kernel
__global__ void boundary_conditions_kernel(
    float* positions,      // [numParticles * 3] - input/output
    float* velocities,     // [numParticles * 3] - input/output
    float restitution,
    int numParticles
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;
    
    float3 pos = make_float3(
        positions[i * 3 + 0],
        positions[i * 3 + 1],
        positions[i * 3 + 2]
    );
    
    float3 vel = make_float3(
        velocities[i * 3 + 0],
        velocities[i * 3 + 1],
        velocities[i * 3 + 2]
    );
    
    bool collision = false;
    
    // if position is outside the box, reflect velocity and clamp position
    if (pos.x < c_boxMin.x) {
        pos.x = c_boxMin.x;
        vel.x = -vel.x * restitution;
        collision = true;
    }

    if (pos.x > c_boxMax.x) {
        pos.x = c_boxMax.x;
        vel.x = -vel.x * restitution;
        collision = true;
    }
    
    if (pos.y < c_boxMin.y) {
        pos.y = c_boxMin.y;
        vel.y = -vel.y * restitution;
        collision = true;
    }
    
    if (pos.y > c_boxMax.y) {
        pos.y = c_boxMax.y;
        vel.y = -vel.y * restitution;
        collision = true;
    }
    
    if (pos.z < c_boxMin.z) {
        pos.z = c_boxMin.z;
        vel.z = -vel.z * restitution;
        collision = true;
    }

    if (pos.z > c_boxMax.z) {
        pos.z = c_boxMax.z;
        vel.z = -vel.z * restitution;
        collision = true;
    }
    
    // update attributes only if there was a collision
    if (collision) {
        positions[i * 3 + 0] = pos.x;
        positions[i * 3 + 1] = pos.y;
        positions[i * 3 + 2] = pos.z;
        
        velocities[i * 3 + 0] = vel.x;
        velocities[i * 3 + 1] = vel.y;
        velocities[i * 3 + 2] = vel.z;
    }
}

// host!!! CPU interface functions
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
    ) {
        cudaMemcpyToSymbol(c_smoothingRadius, &smoothingRadius, sizeof(float));
        
        float smoothingRadius2 = smoothingRadius * smoothingRadius;
        cudaMemcpyToSymbol(c_smoothingRadius2, &smoothingRadius2, sizeof(float));
        cudaMemcpyToSymbol(c_particleMass, &particleMass, sizeof(float));
        cudaMemcpyToSymbol(c_restDensity, &restDensity, sizeof(float));
        cudaMemcpyToSymbol(c_gasConstant, &gasConstant, sizeof(float));
        cudaMemcpyToSymbol(c_viscosity, &viscosity, sizeof(float));
        cudaMemcpyToSymbol(c_dampingFactor, &dampingFactor, sizeof(float));
        cudaMemcpyToSymbol(c_gravity, &gravity, sizeof(float3));
        cudaMemcpyToSymbol(c_boxMin, &boxMin, sizeof(float3));
        cudaMemcpyToSymbol(c_boxMax, &boxMax, sizeof(float3));
    }
    
    void launch_density_pressure_kernel(
        float* d_positions,
        float* d_densities,
        float* d_pressures,
        int numParticles
    ) {
        int blockSize = 256;
        int gridSize = (numParticles + blockSize - 1) / blockSize;
        
        compute_density_pressure_kernel<<<gridSize, blockSize>>>(
            d_positions, d_densities, d_pressures, numParticles
        );
        
        cudaDeviceSynchronize();
    }
    
    void launch_forces_kernel(
        float* d_positions,
        float* d_velocities,
        float* d_densities,
        float* d_pressures,
        float* d_forces,
        int numParticles
    ) {
        int blockSize = 256;
        int gridSize = (numParticles + blockSize - 1) / blockSize;
        
        compute_forces_kernel<<<gridSize, blockSize>>>(
            d_positions, d_velocities, d_densities, d_pressures, d_forces, numParticles
        );
        
        cudaDeviceSynchronize();
    }
    
    void launch_integrate_kernel(
        float* d_positions,
        float* d_velocities,
        float* d_forces,
        float dt,
        int numParticles
    ) {
        int blockSize = 256;
        int gridSize = (numParticles + blockSize - 1) / blockSize;
        
        integrate_kernel<<<gridSize, blockSize>>>(
            d_positions, d_velocities, d_forces, dt, numParticles
        );
        
        cudaDeviceSynchronize();
    }
    
    void launch_boundary_kernel(
        float* d_positions,
        float* d_velocities,
        float restitution,
        int numParticles
    ) {
        int blockSize = 256;
        int gridSize = (numParticles + blockSize - 1) / blockSize;
        
        boundary_conditions_kernel<<<gridSize, blockSize>>>(
            d_positions, d_velocities, restitution, numParticles
        );
        
        cudaDeviceSynchronize();
    }
}