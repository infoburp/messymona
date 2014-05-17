#include "opencv2/opencv.hpp"
#include <png.h>
#include <stdint.h>
#include <cmath>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <ctime>
#include <climits>
 
#define XSIZE 450
#define YSIZE 399
#define NUMPOLYGONS 32
#define NUMSIDES 6
 
typedef struct line {
        float2 v1, v2;
        float m;
        float b;
        __host__ void update() {
                if(v1.x < v2.x) {
                        m = (v2.y - v1.y) / (v2.x - v1.x);
                } else {
                        m = (v1.y - v2.y) / (v1.x - v2.x);
                }
                b = v1.y - m * v1.x;
        }
 
        __device__ bool intersects(int y) {
                if((v1.y > y) != (v2.y > y)) return true;
                return false;
        }
 
        __device__ float findIntersection(int y) {
                return (y - b)/m;
        }
} line;
 
typedef struct polygon {
        line sides[NUMSIDES];
        float transparency;
        uchar3 color;
        float3 colorWithTransparency;
        __host__ polygon() {
                sides[0].v1.x = rand() % XSIZE;
                sides[0].v1.y = rand() % YSIZE;
                sides[0].v2.x = rand() % XSIZE;
                sides[0].v2.y = rand() % YSIZE;
                for(int i = 1; i < NUMSIDES - 1; i++) {
                        sides[i].v1.x = sides[i - 1].v2.x;
                        sides[i].v1.y = sides[i - 1].v2.y;
                        sides[i].v2.x = rand() % XSIZE;
                        sides[i].v2.y = rand() % YSIZE;
                }
                sides[NUMSIDES - 1].v1.x = sides[NUMSIDES - 2].v2.x;
                sides[NUMSIDES - 1].v1.y = sides[NUMSIDES - 2].v2.y;
                sides[NUMSIDES - 1].v2.x = sides[0].v1.x;
                sides[NUMSIDES - 1].v2.y = sides[0].v1.y;
                for(int i = 0; i < NUMSIDES; i++) {
                        sides[i].update();
                }
                transparency = 1;
                for(int i = 0; i < 3; i++) {
                        color.x = rand() % 256;
                        color.y = rand() % 256;
                        color.z = rand() % 256;
                }
                colorWithTransparency.x = color.x * (1 - transparency);
                colorWithTransparency.y = color.y * (1 - transparency);
                colorWithTransparency.z = color.z * (1 - transparency);
        }
        __host__ void mutateGeneral(int factor) {
                if(rand() % 2 == 0) {
                        color.x += ((float)rand() / RAND_MAX) * (256.f / factor) - (256.f / factor / 2);
                        color.y += ((float)rand() / RAND_MAX) * (256.f / factor) - (256.f / factor / 2);
                        color.z += ((float)rand() / RAND_MAX) * (256.f / factor) - (256.f / factor / 2);
                        transparency += ((float)rand() / RAND_MAX) * (1.f / factor) - (1.f / factor / 2);
                        if(transparency > 1) transparency = 1;
                        if(transparency < 0) transparency = 0;
                }
                if(rand() % 2 == 0) {
                        if(rand() % 5 == 0) {
                                sides[0].v1.x += ((float)rand() / RAND_MAX) * ((float)XSIZE / factor) - ((float)XSIZE / factor / 2);
                                sides[0].v1.y += ((float)rand() / RAND_MAX) * ((float)YSIZE / factor) - ((float)YSIZE / factor / 2);
                        }
                        if(rand() % 5 == 0) {
                                sides[0].v2.x += ((float)rand() / RAND_MAX) * ((float)XSIZE / factor) - ((float)XSIZE / factor / 2);
                                sides[0].v2.y += ((float)rand() / RAND_MAX) * ((float)YSIZE / factor) - ((float)YSIZE / factor / 2);
                        }
                        for(int i = 1; i < NUMSIDES - 1; i++) {
                                sides[i].v1.x = sides[i - 1].v2.x;
                                sides[i].v1.y = sides[i - 1].v2.y;
                                if(rand() % 5 == 0) {
                                        sides[i].v2.x += ((float)rand() / RAND_MAX) * ((float)XSIZE / factor) - ((float)XSIZE / factor / 2);
                                        sides[i].v2.y += ((float)rand() / RAND_MAX) * ((float)YSIZE / factor) - ((float)YSIZE / factor / 2);
                                }
                        }
                        sides[NUMSIDES - 1].v1.x = sides[NUMSIDES - 2].v2.x;
                        sides[NUMSIDES - 1].v1.y = sides[NUMSIDES - 2].v2.y;
                        sides[NUMSIDES - 1].v2.x = sides[0].v1.x;
                        sides[NUMSIDES - 1].v2.y = sides[0].v1.y;
                }
                for(int i = 0; i < NUMSIDES; i++) {
                        sides[i].update();
                }
                colorWithTransparency.x = color.x * (1 - transparency);
                colorWithTransparency.y = color.y * (1 - transparency);
                colorWithTransparency.z = color.z * (1 - transparency);
        }
        __host__ void mutateColor(int factor) {
                color.x += ((float)rand() / RAND_MAX) * (256.f / factor) - (256.f / factor / 2);
                color.y += ((float)rand() / RAND_MAX) * (256.f / factor) - (256.f / factor / 2);
                color.z += ((float)rand() / RAND_MAX) * (256.f / factor) - (256.f / factor / 2);
                transparency += ((float)rand() / RAND_MAX) * (1.f / factor) - (1.f / factor / 2);
                if(transparency > 1) transparency = 1;
                if(transparency < 0) transparency = 0;
 
                colorWithTransparency.x = color.x * (1 - transparency);
                colorWithTransparency.y = color.y * (1 - transparency);
                colorWithTransparency.z = color.z * (1 - transparency);
        }
        __host__ void mutateShape(int factor) {
                if(rand() % 5 == 0) {
                        sides[0].v1.x += ((float)rand() / RAND_MAX) * ((float)XSIZE / factor) - ((float)XSIZE / factor / 2);
                        sides[0].v1.y += ((float)rand() / RAND_MAX) * ((float)YSIZE / factor) - ((float)YSIZE / factor / 2);
                }
                if(rand() % 5 == 0) {
                        sides[0].v2.x += ((float)rand() / RAND_MAX) * ((float)XSIZE / factor) - ((float)XSIZE / factor / 2);
                        sides[0].v2.y += ((float)rand() / RAND_MAX) * ((float)YSIZE / factor) - ((float)YSIZE / factor / 2);
                }
                for(int i = 1; i < NUMSIDES - 1; i++) {
                        sides[i].v1.x = sides[i - 1].v2.x;
                        sides[i].v1.y = sides[i - 1].v2.y;
                        if(rand() % 5 == 0) {
                                sides[i].v2.x += ((float)rand() / RAND_MAX) * ((float)XSIZE / factor) - ((float)XSIZE / factor / 2);
                                sides[i].v2.y += ((float)rand() / RAND_MAX) * ((float)YSIZE / factor) - ((float)YSIZE / factor / 2);
                        }
                }
                sides[NUMSIDES - 1].v1.x = sides[NUMSIDES - 2].v2.x;
                sides[NUMSIDES - 1].v1.y = sides[NUMSIDES - 2].v2.y;
                sides[NUMSIDES - 1].v2.x = sides[0].v1.x;
                sides[NUMSIDES - 1].v2.y = sides[0].v1.y;
        }
} polygon;
 
typedef struct genome {
        polygon polygons[NUMPOLYGONS];
        int score;
        __host__ void mutateGeneral(int i, int factor, int p1, int p2) {
                score = 0;
                polygons[i].mutateGeneral(factor);
                if(rand() % 10 == 0) {
                        polygon tmp = polygons[p1];
                        polygons[p1] = polygons[p2];
                        polygons[p2] = tmp;
                }
        }
        __host__ void mutateColor(int i, int factor) {
                score = 0;
                polygons[i].mutateColor(factor);
        }
        __host__ void mutateShape(int i, int factor) {
                score = 0;
                polygons[i].mutateShape(factor);
        }
        __host__ void mutateOrder(int p1, int p2) {
                polygon tmp = polygons[p1];
                polygons[p1] = polygons[p2];
                polygons[p2] = tmp;
        }
} genome;
 
__global__ void findIntersections(genome* g, float* dev_intersections) {
        int y = blockIdx.x;
        int poly = threadIdx.x;
 
        if(y >= YSIZE) return;
 
        float intersections[NUMSIDES];
        for(int side = 0; side < NUMSIDES; side++) {
                intersections[side] = 0;
                if(!g->polygons[poly].sides[side].intersects(y)) continue;
                intersections[side] = g->polygons[poly].sides[side].findIntersection(y);
        }
        for(int side = 0; side < NUMSIDES; side++) {
                int minIndex = side;
                for(int j = side + 1; j < NUMSIDES; j++) {
                        if(intersections[j] < intersections[minIndex]) minIndex = j;
                }
                float tmp = intersections[side];
                intersections[side] = intersections[minIndex];
                intersections[minIndex] = tmp;
        }
        int offset = y * NUMPOLYGONS * NUMSIDES + poly * NUMSIDES;
        for(int i = 0; i < NUMSIDES; i++) {
                dev_intersections[offset + i] = intersections[i];
        }
}
 
__global__ void draw(genome* g, uchar* dev_pixels, float* dev_intersections) {
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;
 
        if(y >= YSIZE || x > XSIZE) return;
 
        __shared__ float intersections[NUMPOLYGONS * NUMSIDES];
        {
                int offset = y * NUMPOLYGONS * NUMSIDES;
                for(int xPrime = x - blockIdx.x * blockDim.x; xPrime < NUMPOLYGONS * NUMSIDES; xPrime += XSIZE / blockDim.x) {
                        intersections[xPrime] = dev_intersections[offset + xPrime];
                }
        }
        __syncthreads();
 
        uchar3 color = make_uchar3(255, 255, 255);
        for(int poly = 0; poly < NUMPOLYGONS; poly++) {
                int offset = poly * NUMSIDES;
                for(int side = 0; side < NUMSIDES; side += 2) {
                        if(x > intersections[offset + side] && x < intersections[offset + side + 1]) {
                                color.x = color.x * g->polygons[poly].transparency + g->polygons[poly].colorWithTransparency.x;
                                color.y = color.y * g->polygons[poly].transparency + g->polygons[poly].colorWithTransparency.y;
                                color.z = color.z * g->polygons[poly].transparency + g->polygons[poly].colorWithTransparency.z;
                                break;
                        }
                }
        }
 
        int offset = (x + y * XSIZE) * 3;
        dev_pixels[offset] = color.x;
        dev_pixels[offset + 1] = color.y;
        dev_pixels[offset + 2] = color.z;
}
 
__global__ void score(genome* g, uchar* dev_reference, float* dev_intersections) {
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;
 
        if(y >= YSIZE || x > XSIZE) return;
 
        __shared__ float intersections[NUMPOLYGONS * NUMSIDES];
        {
                int offset = y * NUMPOLYGONS * NUMSIDES;
                for(int xPrime = x - blockIdx.x * blockDim.x; xPrime < NUMPOLYGONS * NUMSIDES; xPrime += blockDim.x) {
                        intersections[xPrime] = dev_intersections[offset + xPrime];
                }
        }
        __syncthreads();
 
        uchar3 color = make_uchar3(255, 255, 255);
        for(int poly = 0; poly < NUMPOLYGONS; poly++) {
                int offset = poly * NUMSIDES;
                for(int side =  0; side < NUMSIDES; side += 2) {
                        if(x > intersections[offset + side] && x < intersections[offset + side + 1]) {
                                color.x = color.x * g->polygons[poly].transparency + g->polygons[poly].colorWithTransparency.x;
                                color.y = color.y * g->polygons[poly].transparency + g->polygons[poly].colorWithTransparency.y;
                                color.z = color.z * g->polygons[poly].transparency + g->polygons[poly].colorWithTransparency.z;
                                break;
                        }
                }
        }
 
        int score = 0;
        int offset = (x + y * XSIZE) * 3;
        score += (dev_reference[offset] - color.x) * (dev_reference[offset] - color.x);
        score += (dev_reference[offset + 1] - color.y) * (dev_reference[offset + 1] - color.y);
        score += (dev_reference[offset + 2] - color.z) * (dev_reference[offset + 2] - color.z);
        score /= 256;
 
        atomicAdd(&g->score, score);
}
 
void writeImage(uchar* dev_pixels, const char* name) {
 
        FILE *fp = fopen(name, "wb");
        uchar* pixels = (uchar*)malloc(sizeof(uchar) * XSIZE * YSIZE * 3);
        cudaMemcpy(pixels, dev_pixels, sizeof(uchar) * XSIZE * YSIZE * 3, cudaMemcpyDeviceToHost);
 
        png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
        png_infop info_ptr = png_create_info_struct(png_ptr);
        png_init_io(png_ptr, fp);
        png_set_IHDR(png_ptr, info_ptr, XSIZE, YSIZE,
                        8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                        PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);
        png_write_info(png_ptr, info_ptr);
 
        png_byte** row_pointers = (png_byte**)png_malloc(png_ptr, YSIZE * sizeof (png_byte *));
 
        for (int y = 0; y < YSIZE; ++y) {
                png_byte *row = (png_byte*)png_malloc(png_ptr, sizeof(uint8_t) * XSIZE * 3);
                row_pointers[y] = row;
                for (int x = 0; x < XSIZE; ++x) {
                        int offset = (x + y * XSIZE) * 3;
                        *row++ = pixels[offset + 2];
                        *row++ = pixels[offset + 1];
                        *row++ = pixels[offset];
                }
        }
 
        png_init_io (png_ptr, fp);
        png_set_rows (png_ptr, info_ptr, row_pointers);
        png_write_png (png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, NULL);
        png_write_end(png_ptr, NULL);
 
        for(int y = 0; y < YSIZE; y++) {
                free(row_pointers[y]);
        }
        free(pixels);
        free(row_pointers);
        fclose(fp);
}
 
int main(int argc, char** argv) {
 
        srand(time(NULL));
 
        cv::Mat reference = cv::imread("/home/peter/Desktop/image.png");
 
        uchar* dev_reference;
        cudaMalloc(&dev_reference, sizeof(uchar) * XSIZE * YSIZE * 3);
        cudaMemcpy(dev_reference, (uchar*)reference.data, sizeof(uchar) * XSIZE * YSIZE * 3, cudaMemcpyHostToDevice);
 
        genome host_genome;
        host_genome.score = INT_MAX;
 
        genome* dev_genome;
        cudaMalloc(&dev_genome, sizeof(genome));
 
        genome mutant;
        mutant = host_genome;
        mutant.score = 0;
 
        cudaMemcpy(dev_genome, &mutant, sizeof(genome), cudaMemcpyHostToDevice);
 
        dim3 grid_dim(1, YSIZE);
        dim3 block_dim(XSIZE, 1);
 
        float* dev_intersections;
        cudaMalloc(&dev_intersections, sizeof(float) * NUMPOLYGONS * NUMSIDES * YSIZE);
 
        for(int generation = 0; generation < 1000000; generation++) {
                printf("   %d | %f \n", generation, (float)host_genome.score / XSIZE / YSIZE / 256 / 3);
 
                cudaMemcpy(dev_genome, &mutant, sizeof(genome), cudaMemcpyHostToDevice);
                findIntersections<<<YSIZE, NUMPOLYGONS>>>(dev_genome, dev_intersections);
                score<<<grid_dim, block_dim>>>(dev_genome, dev_reference, dev_intersections);
                cudaMemcpy(&mutant, dev_genome, sizeof(genome), cudaMemcpyDeviceToHost);
 
                if(mutant.score < host_genome.score && mutant.score > 0) {
                        host_genome = mutant;
 
                        uchar* dev_pixels;
                        cudaMalloc(&dev_pixels, sizeof(uchar) * XSIZE * YSIZE * 3 );
                        cudaMemcpy(dev_genome, &host_genome, sizeof(genome), cudaMemcpyHostToDevice);
 
                        findIntersections<<<YSIZE, NUMPOLYGONS>>>(dev_genome, dev_intersections);
                        draw<<<grid_dim, block_dim>>>(dev_genome, dev_pixels, dev_intersections);
 
                        std::stringstream ss;
                        ss << "/home/peter/Desktop/images/image";
                        ss << generation;
                        ss << ".png";
 
                        writeImage(dev_pixels, ss.str().c_str());
                        cudaFree(dev_pixels);
                }
 
                mutant = host_genome;
 
                mutant.mutateGeneral(generation % NUMPOLYGONS + 1, 3, rand() % NUMPOLYGONS, rand() % NUMPOLYGONS);
        }
 
        cudaFree(dev_reference);
        cudaFree(dev_genome);
 
        return 0;
}
