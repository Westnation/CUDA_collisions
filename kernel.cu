#include <curand.h>
#include <curand_kernel.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <vector>
#include <time.h>
#include <fstream>

class Vector2D
{
public:
    double x, y;

    __host__ __device__ Vector2D()
    {
        x = y = 0;
    }

    __host__ __device__ Vector2D(double x, double y)
    {
        this->x = x;
        this->y = y;
    }

    __host__ __device__ double magnitude()
    {
        return sqrt(pow(x, 2) + pow(y, 2));
    }

    __host__ __device__ Vector2D normalized()
    {
        double mag = magnitude();
        if (mag == 0) return Vector2D();
        double x = sqrt(abs(this->x) / mag);
        double y = sqrt(abs(this->y) / mag);
        return Vector2D(x, y);
    }

    __host__ __device__ Vector2D normal()
    {
        return Vector2D(-this->y, this->x).normalized();
    }

    __host__ __device__ double dot(Vector2D vector)
    {
        return this->x * vector.x + this->y + vector.y;
    }

    __host__ __device__ Vector2D& operator+=(const Vector2D& rightVector)
    {
        this->x += rightVector.x;
        this->y += rightVector.y;
        return *this;
    }

    __host__ __device__ Vector2D& operator-=(const Vector2D& rightVector)
    {
        this->x -= rightVector.x;
        this->y -= rightVector.y;
        return *this;
    }

    __host__ __device__ bool operator==(const Vector2D& rightVector)
    {
        return (this->x == rightVector.x && this->y == rightVector.y);
    }

    __host__ __device__ Vector2D operator*(double velocity)
    {
        return Vector2D(x * velocity, y * velocity);
    }

    __host__ __device__ Vector2D operator+(const Vector2D& rightVector)
    {
        return Vector2D(x + rightVector.x, y + rightVector.y);
    }

    __host__ __device__ Vector2D operator-(const Vector2D& rightVector)
    {
        return Vector2D(x - rightVector.x, y - rightVector.y);
    }
    __host__ __device__ Vector2D operator-()
    {
        return Vector2D(-x, -y);
    }
};

class Body
{
public:
    Vector2D position, velocity;
    double mass;
    double inverseMass;
    double radius;
    bool canDivide;

    __host__ __device__ Body(Vector2D position, Vector2D velocity, double mass, double raduis, bool canDivide)
    {
        this->position = position;
        this->velocity = velocity;
        this->mass = mass;
        this->radius = raduis;
        this->canDivide = canDivide;

        if (mass == 0) inverseMass = 0;
        else inverseMass = 1 / mass;
    }

};

void generateBodies(std::vector<Body>* bodies, unsigned long long numberOfBodies);
void serial(std::vector<Body> bodies, double time, double percent);
void parallel(Body* bodies, double time, double percent);

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

        Body currentBody = Body(position, velocity, mass, radius, true);
        (*bodies).push_back(currentBody);
    }
}

void serial(std::vector<Body>* bodies, double time, double percent)
{
    double currentTime = 0;
    double deltaTime = 0.02; // шаг симуляции

    double elasticity = 1; // эластичность 1 - полное упругое столкновение

    while (currentTime <= time)
    {
        for (unsigned long long i = 0; i < (*bodies).size(); i++)
        {
            for (unsigned long long j = 0; j < (*bodies).size(); j++)
            {
                if (i == j) continue;

                // определяем текущую дистанцию проникновения (отрицательная - проникновения нет, 0 - касание)
                Vector2D distance = (*bodies)[i].position - (*bodies)[j].position;
                double penetrationDepth = (*bodies)[i].radius + (*bodies)[j].radius - distance.magnitude();

                // если есть проникновение
                if (penetrationDepth > 0)
                {
                    // устраняем overlapping-ситуацию - пересечение двух тел
                    Vector2D penetrationResolution = distance.normalized() * (penetrationDepth / ((*bodies)[i].inverseMass + (*bodies)[j].inverseMass));
                    (*bodies)[i].position += (penetrationResolution * (*bodies)[i].inverseMass);
                    (*bodies)[j].position -= (penetrationResolution * (*bodies)[j].inverseMass);

                    // вычисление разделяющей скорости
                    Vector2D normal = ((*bodies)[i].position - (*bodies)[j].position).normalized();
                    Vector2D relativeVelocityVector = (*bodies)[i].velocity - (*bodies)[j].velocity;
                    double separationVelocity = relativeVelocityVector.dot(normal);
                    double newSeparationVelocity = -separationVelocity * elasticity;

                    // получение вектора импульса
                    double separationVelocityDifference = newSeparationVelocity - separationVelocity;
                    double impulse = separationVelocityDifference / ((*bodies)[i].inverseMass + (*bodies)[j].inverseMass);
                    Vector2D impulseVector = normal * impulse;

                    // изменение скоростей тел
                    (*bodies)[i].velocity += (impulseVector * (*bodies)[i].inverseMass);
                    (*bodies)[j].velocity -= (impulseVector * (*bodies)[j].inverseMass);

                    // шанс разделения соприкоснувшихся объектов пополам
                    if ((*bodies)[i].canDivide) // может ли текущее тело делится
                    {
                        double chance1 = double(rand()) / double(RAND_MAX);
                        if (chance1 < percent) // если тело разделилось
                        {
                            Vector2D position = (*bodies)[i].position - normal.normal() * ((*bodies)[i].radius / 2);
                            Body body = Body(position, -(*bodies)[i].velocity, (*bodies)[i].mass / 4, (*bodies)[i].radius / 4, false);
                            (*bodies)[i].position = (*bodies)[i].position + normal.normal() * ((*bodies)[i].radius / 2);
                            (*bodies)[i].mass /= 4;
                            (*bodies)[i].radius /= 4;
                            (*bodies)[i].canDivide = false;
                            (*bodies).push_back(body);
                        }
                    }

                    if ((*bodies)[j].canDivide) // может ли текущее тело делится
                    {
                        double chance2 = double(rand()) / double(RAND_MAX);
                        if (chance2 < percent) // если тело разделилось
                        {
                            Vector2D position = (*bodies)[j].position - normal.normal() * ((*bodies)[j].radius / 2);
                            Body body = Body(position, -(*bodies)[j].velocity, (*bodies)[j].mass / 4, (*bodies)[j].radius / 4, false);
                            (*bodies)[j].position = (*bodies)[j].position + normal.normal() * ((*bodies)[j].radius / 2);
                            (*bodies)[j].mass /= 4;
                            (*bodies)[j].radius /= 4;
                            (*bodies)[j].canDivide = false;
                            (*bodies).push_back(body);
                        }
                    }
                }
            }
        }

        // изменение позиции тел
        for (unsigned long long i = 0; i < (*bodies).size(); i++)
        {
            (*bodies)[i].position += ((*bodies)[i].velocity * deltaTime);
        }

        currentTime += deltaTime;
    }
}

__global__ void parallelCollision(Body* bodies, unsigned long long* size, double percent, curandState* state, unsigned long long seed, unsigned long long blockCount)
{
    unsigned long long tid = threadIdx.x + blockIdx.x * blockDim.x;

    double elasticity = 1; // эластичность 1 - полное упругое столкновение
    while (tid < *size)
    {
        for (unsigned long long j = tid + 1; j < *size; j++)
        {
            // определяем текущую дистанцию проникновения (отрицательная - проникновения нет, 0 - касание)
            Vector2D distance = bodies[tid].position - bodies[j].position;
            double penetrationDepth = bodies[tid].radius + bodies[j].radius - distance.magnitude();
            // если есть проникновение
            if (penetrationDepth > 0 && tid < j)
            {
                // устраняем overlapping-ситуацию - пересечение двух тел
                Vector2D penetrationResolution = distance.normalized() * (penetrationDepth / (bodies[tid].inverseMass + bodies[j].inverseMass));
                bodies[tid].position += (penetrationResolution * bodies[tid].inverseMass);
                bodies[j].position -= (penetrationResolution * bodies[j].inverseMass);

                // вычисление разделяющей скорости
                Vector2D normal = (bodies[tid].position - bodies[j].position).normalized();
                Vector2D relativeVelocityVector = bodies[tid].velocity - bodies[j].velocity;
                double separationVelocity = relativeVelocityVector.dot(normal);
                double newSeparationVelocity = -separationVelocity * elasticity;

                // получение вектора импульса
                double separationVelocityDifference = newSeparationVelocity - separationVelocity;
                double impulse = separationVelocityDifference / (bodies[tid].inverseMass + bodies[j].inverseMass);
                Vector2D impulseVector = normal * impulse;

                // изменение скоростей тел
                bodies[tid].velocity += (impulseVector * bodies[tid].inverseMass);
                bodies[j].velocity -= (impulseVector * bodies[j].inverseMass);

                // шанс разделения соприкоснувшихся объектов пополам
                if (bodies[tid].canDivide) // может ли текущее тело делится
                {
                    curand_init(tid + seed, 0, 0, &state[0]);
                    double cur_rand = curand_uniform(&state[0]);
                    double chance1 = cur_rand / double(RAND_MAX);
                    if (chance1 < percent) // если тело разделилось
                    {
                        Vector2D position = bodies[tid].position - normal.normal() * (bodies[tid].radius / 2);
                        Body body = Body(position, -bodies[tid].velocity, bodies[tid].mass / 4, bodies[tid].radius / 4, false);
                        bodies[tid].position = bodies[tid].position + normal.normal() * (bodies[tid].radius / 2);
                        bodies[tid].mass /= 4;
                        bodies[tid].radius /= 4;
                        bodies[tid].canDivide = false;
                        atomicAdd(size, 1);
                        bodies[*size - 1] = body;
                    }
                }

                if (bodies[j].canDivide) // может ли текущее тело делится
                {
                    curand_init(j + seed, 0, 0, &state[0]);
                    double cur_rand = curand_uniform(&state[0]);
                    double chance2 = cur_rand / double(RAND_MAX);
                    if (chance2 < percent) // если тело разделилось
                    {
                        Vector2D position = bodies[j].position - normal.normal() * (bodies[j].radius / 2);
                        Body body = Body(position, -bodies[j].velocity, bodies[j].mass / 4, bodies[j].radius / 4, false);
                        bodies[j].position = bodies[j].position + normal.normal() * (bodies[j].radius / 2);
                        bodies[j].mass /= 4;
                        bodies[j].radius /= 4;
                        bodies[j].canDivide = false;
                        atomicAdd(size, 1);
                        bodies[*size - 1] = body;
                    }
                }
            }
        }
        tid += (blockCount * blockDim.x);
    }
}

__global__ void parallelPositions(Body* bodies, unsigned long long *size, unsigned long long blockCount)
{
    unsigned long long tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < *size)
    {
        bodies[tid].position += bodies[tid].velocity * 0.02;
        tid += (blockCount * blockDim.x);
    }
}

int main()
{
    setlocale(LC_ALL, "Russian");

    // количество тел
    unsigned long long numberOfBodies;
    printf("Введите количество тел (целое): ");
    std::cin >> numberOfBodies;

    // хранение тел
    std::vector<Body> bodies;
    generateBodies(&bodies, numberOfBodies);
    std::vector<Body> serialBodies;
    serialBodies.assign(bodies.begin(), bodies.end());

    // вероятность разделения
    double percent;
    printf("Введите вероятность разделения тела (вещественное от 0 до 1). Для отключения разделения введите 0: ");
    std::cin >> percent;

    // время симуляции
    double time;
    printf("Введите время симуляции (вещественное): ");
    std::cin >> time;

    // последовательное вычисление
    printf("\nПОСЛЕДОВАТЕЛЬНОЕ ВЫЧИСЛЕНИЕ...\n");
    time_t start = clock();
    serial(&serialBodies, time, percent);
    printf("Время последовательного вычисления: %f\n", (double)(clock() - start) / CLOCKS_PER_SEC);

    // запись результатов последовательного вычисления в файл
    std::ofstream serialResults ("serialResults.txt");
    for (unsigned long long i = 0; i < serialBodies.size(); i++)
    {
        serialResults << i << " " << serialBodies[i].position.x << " " << serialBodies[i].position.y << "\n";
    }
    serialResults.close();

    unsigned long long size = bodies.size();
    unsigned long long *size_dev;
    
    // параметры текущего устройства
    int dev = 0;
    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    int threads = deviceProp.maxThreadsPerBlock;
    int blocks = (size + threads - 1) / threads;

    // параллельное вычисление
    printf("\nПАРАЛЛЕЛЬНОЕ ВЫЧИСЛЕНИЕ...\n");
    start = clock();

    // выделение памяти на видеокарте и копирование туда массива тел и его размер
    Body* dev_bodies;
    cudaMalloc((void**)&dev_bodies, 2 * size * sizeof(Body));
    cudaMemcpy(dev_bodies, bodies.data(), size * sizeof(Body), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&size_dev, sizeof(unsigned long long));
    cudaMemcpy(size_dev, &size, sizeof(unsigned long long), cudaMemcpyHostToDevice);

    // curand - получение случайных чисел на видеокарте
    curandState* d_state;
    cudaMalloc((void**)&d_state, sizeof(curandState));

    // основной цикл симуляции
    double deltaTime = 0.02;
    for (double t = 0; t <= time; t += deltaTime)
    {
        parallelCollision <<< blocks, threads >>> (dev_bodies, size_dev, percent, d_state, clock(), blocks);
        cudaDeviceSynchronize();
        parallelPositions <<< blocks, threads >>> (dev_bodies, size_dev, blocks);
        cudaDeviceSynchronize();
    }
    printf("Время параллельного вычисления: %f\n", (double)(clock() - start) / CLOCKS_PER_SEC);

    // копирование результатов вычисления с ГПУ
    cudaMemcpy(&size, size_dev, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    Body* parallelBodies = (Body*)malloc(sizeof(Body) * size);
    cudaMemcpy(parallelBodies, dev_bodies, size * sizeof(Body), cudaMemcpyDeviceToHost);

    // запись результатов параллельного вычисления в файл
    std::ofstream parallelResults("parallelResults.txt");
    for (unsigned long long i = 0; i < size; i++)
    {
        parallelResults << i << " " << parallelBodies[i].position.x << " " << parallelBodies[i].position.y << "\n";
    }
    parallelResults.close();

    // освобождение памяти на ГПУ
    cudaFree(d_state);
    cudaFree(size_dev);
    cudaFree(dev_bodies);

    // вывод результатов
    unsigned long long max_size = serialBodies.size() > size ? serialBodies.size() : size;
    unsigned long long min_size = serialBodies.size() < size ? serialBodies.size() : size;
    unsigned long long counter = 0;
    int index1;
    for (unsigned long long i = 0; i < min_size; i++)
    {
        if (serialBodies[i].position == parallelBodies[i].position) counter++;
        else index1 = i;
    }
    printf("\nРЕЗУЛЬТАТЫ:\n");
    if (percent != 0 && percent != 1) printf("ВНИМАНИЕ: заданный шанс деления тел не равен 0 или 1. Результаты двух методов могут отличаться.\n");
    printf("Количество тел в результате последовательного вычисления: %llu\n", serialBodies.size());
    printf("Количество тел в результате параллельного вычисления: %llu\n", size);
    printf("Совпадений позиций тел: %llu из %llu. В процентном соотношении: %3.4f\n", counter, max_size, 100 * (double)counter / max_size);
    printf("Были созданы 2 файла: serialResults.txt и parallelResults.txt. В них записаны конечные позиции тел.");
    printf("%d", index1);
}