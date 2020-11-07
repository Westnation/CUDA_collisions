#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <vector>
#include <time.h>

class Vector2D
{
public:
    double x, y;

    Vector2D()
    {
        x = y = 0;
    }

    Vector2D(double x, double y)
    {
        this->x = x;
        this->y = y;
    }

    double magnitude()
    {
        return sqrt(pow(x, 2) + pow(y, 2));
    }

    Vector2D normalized()
    {
        if (magnitude() == 0) return Vector2D();
        double x = sqrt(this->x / magnitude());
        double y = sqrt(this->y / magnitude());
        return Vector2D(x, y);
    }

    Vector2D normal()
    {
        return Vector2D(-this->y, this->x).normalized();
    }

    double dot(Vector2D vector)
    {
        return this->x * vector.x + this->y + vector.y;
    }

    Vector2D& operator+=(const Vector2D& rightVector)
    {
        this->x += rightVector.x;
        this->y += rightVector.y;
        return *this;
    }

    Vector2D& operator-=(const Vector2D& rightVector)
    {
        this->x -= rightVector.x;
        this->y -= rightVector.y;
        return *this;
    }

    Vector2D operator*(double velocity)
    {
        return Vector2D(x * velocity, y * velocity);
    }

    Vector2D operator-(const Vector2D& rightVector)
    {
        return Vector2D(x - rightVector.x, y - rightVector.y);
    }
};

class Body
{
public:
    Vector2D position, velocity;
    double mass;
    double inverseMass;
    double radius;

    Body(Vector2D position, Vector2D velocity, double mass, double raduis)
    {
        this->position = position;
        this->velocity = velocity;
        this->mass = mass;
        this->radius = raduis;

        if (mass == 0) inverseMass = 0;
        else inverseMass = 1 / mass;
    }
};

void generateBodies(std::vector<Body>* bodies, unsigned long long numberOfBodies);
void serial(std::vector<Body> bodies, double time);
void parallel();

void generateBodies(std::vector<Body>* bodies, unsigned long long numberOfBodies)
{
    for (unsigned long long i = 0; i < numberOfBodies; i++)
    {
        double lower_bound_position = -5000;
        double upper_bound_position = 5000;

        double lower_bound_velocity = -10;
        double upper_bound_velocity = 10;

        double xPosition = ((double(rand()) / double(RAND_MAX)) * (upper_bound_position - lower_bound_position)) + lower_bound_position;
        double yPosition = ((double(rand()) / double(RAND_MAX)) * (upper_bound_position - lower_bound_position)) + lower_bound_position;
        Vector2D position = Vector2D(xPosition, yPosition);

        double radius = double(rand()) / double(RAND_MAX) * 5;
        double mass = double(rand()) / double(RAND_MAX) * 5;

        double xDirection = ((double(rand()) / double(RAND_MAX)) * (upper_bound_velocity - lower_bound_velocity)) + lower_bound_velocity;
        double yDirection = ((double(rand()) / double(RAND_MAX)) * (upper_bound_velocity - lower_bound_velocity)) + lower_bound_velocity;
        Vector2D velocity = Vector2D(xDirection, yDirection);

        Body currentBody = Body(position, velocity, mass, radius);
        (*bodies).push_back(currentBody);
    }
}

void serial(std::vector<Body> bodies, double time)
{
    double currentTime = 0;
    double deltaTime = 0.02; // шаг симул€ции

    double elasticity = 1; // эластичность 1 - полное упругое столкновение

    while (currentTime < time)
    {
        for (unsigned long long i = 0; i < bodies.size(); i++)
        {
            for (unsigned long long j = 0; j < bodies.size(); j++)
            {
                if (i == j) continue;

                // определ€ем текущую дистанцию проникновени€ (отрицательна€ - проникновени€ нет, 0 - касание)
                Vector2D distance = bodies[i].position - bodies[j].position;
                double penetrationDepth = bodies[i].radius + bodies[j].radius - distance.magnitude();

                // если есть проникновение
                if (penetrationDepth > 0)
                {
                    // устран€ем overlapping-ситуацию - пересечение двух тел
                    Vector2D penetrationResolution = distance.normalized() * (penetrationDepth / (bodies[i].inverseMass + bodies[j].inverseMass));
                    bodies[i].position += (penetrationResolution * bodies[i].inverseMass);
                    bodies[j].position -= (penetrationResolution * bodies[j].inverseMass);

                    // вычисление раздел€ющей скорости
                    Vector2D normal = (bodies[i].position - bodies[j].position).normalized();
                    Vector2D relativeVelocityVector = bodies[i].velocity - bodies[j].velocity;
                    double separationVelocity = relativeVelocityVector.dot(normal);
                    double newSeparationVelocity = -separationVelocity * elasticity;

                    // получение вектора импульса
                    double separationVelocityDifference = newSeparationVelocity - separationVelocity;
                    double impulse = separationVelocityDifference / (bodies[i].inverseMass + bodies[j].inverseMass);
                    Vector2D impulseVector = normal * impulse;

                    // изменение скоростей тел
                    bodies[i].velocity += (impulseVector * bodies[i].inverseMass);
                    bodies[j].velocity -= (impulseVector * bodies[j].inverseMass);

                    std::cout << currentTime << "\n";
                }
            }
        }

        // изменение позиции тел
        for (unsigned long long i = 0; i < bodies.size(); i++)
        {
            bodies[i].position += (bodies[i].velocity * deltaTime);
        }

        currentTime += deltaTime;
    }
}

int main()
{
    setlocale(LC_ALL, "Russian");

    unsigned long long numberOfBodies;
    printf("¬ведите количество тел (целое): ");
    std::cin >> numberOfBodies;

    std::vector<Body> bodies;
    generateBodies(&bodies, numberOfBodies);
    std::vector<Body> currentBodies(bodies);

    double percent;
    printf("¬ведите веро€тность разделени€ тела (вещественное от 0 до 1). ƒл€ отключени€ разделени€ введите 0: ");
    std::cin >> percent;

    double time;
    printf("¬ведите врем€ симул€ции (вещественное): ");
    std::cin >> time;

    printf("ѕќ—Ћ≈ƒќ¬ј“≈Ћ№Ќќ≈ ¬џ„»—Ћ≈Ќ»≈...\n");
    time_t start = clock();
    serial(currentBodies, time);
    printf("¬рем€ последовательного вычислени€: %f\n", (double)(clock() - start) / CLOCKS_PER_SEC);
}

//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//
//#include <stdio.h>
//
//cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);
//
//__global__ void addKernel(int* c, const int* a, const int* b)
//{
//	int i = threadIdx.x;
//	c[i] = a[i] + b[i];
//}
//
//int main()
//{
//	const int arraySize = 5;
//	const int a[arraySize] = { 1, 2, 3, 4, 5 };
//	const int b[arraySize] = { 10, 20, 30, 40, 50 };
//	int c[arraySize] = { 0 };
//
//	// Add vectors in parallel.
//	cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "addWithCuda failed!");
//		return 1;
//	}
//
//	printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
//		c[0], c[1], c[2], c[3], c[4]);
//
//	// cudaDeviceReset must be called before exiting in order for profiling and
//	// tracing tools such as Nsight and Visual Profiler to show complete traces.
//	cudaStatus = cudaDeviceReset();
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaDeviceReset failed!");
//		return 1;
//	}
//
//	return 0;
//}
//
//// Helper function for using CUDA to add vectors in parallel.
//cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size)
//{
//	int* dev_a = 0;
//	int* dev_b = 0;
//	int* dev_c = 0;
//	cudaError_t cudaStatus;
//
//	// Choose which GPU to run on, change this on a multi-GPU system.
//	cudaStatus = cudaSetDevice(0);
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
//		goto Error;
//	}
//
//	// Allocate GPU buffers for three vectors (two input, one output)    .
//	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMalloc failed!");
//		goto Error;
//	}
//
//	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMalloc failed!");
//		goto Error;
//	}
//
//	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMalloc failed!");
//		goto Error;
//	}
//
//	// Copy input vectors from host memory to GPU buffers.
//	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMemcpy failed!");
//		goto Error;
//	}
//
//	cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMemcpy failed!");
//		goto Error;
//	}
//
//	// Launch a kernel on the GPU with one thread for each element.
//	addKernel << <1, size >> > (dev_c, dev_a, dev_b);
//
//	// Check for any errors launching the kernel
//	cudaStatus = cudaGetLastError();
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
//		goto Error;
//	}
//
//	// cudaDeviceSynchronize waits for the kernel to finish, and returns
//	// any errors encountered during the launch.
//	cudaStatus = cudaDeviceSynchronize();
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
//		goto Error;
//	}
//
//	// Copy output vector from GPU buffer to host memory.
//	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMemcpy failed!");
//		goto Error;
//	}
//
//Error:
//	cudaFree(dev_c);
//	cudaFree(dev_a);
//	cudaFree(dev_b);
//
//	return cudaStatus;
//}
