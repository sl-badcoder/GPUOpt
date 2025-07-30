#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define CHECK(err,msg) \
  if ((err) != CL_SUCCESS) { fprintf(stderr, "%s failed (%d)\n", msg, err); exit(1); }

int bitonic_sort_opencl(int *arr, size_t size, size_t start_k) {
    cl_int err;
    cl_platform_id   pf;
    cl_device_id     dv;
    cl_context       ctx;
    cl_command_queue cq;
    cl_program       prog;
    cl_kernel        krn;
    cl_mem           d_arr;
    cl_event         ev;

    CHECK(clGetPlatformIDs(1, &pf, NULL),               "clGetPlatformIDs");
    CHECK(clGetDeviceIDs(pf, CL_DEVICE_TYPE_GPU, 1, &dv, NULL),
          "clGetDeviceIDs");

    ctx = clCreateContext(NULL, 1, &dv, NULL, NULL, &err);   CHECK(err,"clCreateContext");
    cq  = clCreateCommandQueue(ctx, dv, 0, &err);            CHECK(err,"clCreateCommandQueue");

    FILE *f = fopen("bitonic_opencl.cl","r");
    if (!f) { perror("fopen"); return -1; }
    fseek(f,0,SEEK_END);
    size_t sz = ftell(f);
    rewind(f);
    char *code = malloc(sz+1);
    fread(code,1,sz,f);
    code[sz]='\0';
    fclose(f);

    prog = clCreateProgramWithSource(ctx, 1, (const char**)&code, NULL, &err);
    free(code);
    CHECK(err,"clCreateProgramWithSource");

    clBuildProgram(prog, 1, &dv, NULL, NULL, NULL);


    krn = clCreateKernel(prog, "globalSwap", &err);  CHECK(err, "clCreateKernel");

    d_arr = clCreateBuffer(ctx,
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        size * sizeof(int),
        arr, &err);
    CHECK(err,"clCreateBuffer");

    int start_i = 0;
    while ((1u << start_i) < start_k && (1u << start_i) < size)
        ++start_i;

    size_t block_size = 256;
    size_t num_blocks = (size + block_size - 1) / block_size;
    size_t gws = num_blocks * block_size;
    size_t lws = block_size;


    for (int i = start_i; (1u << i) <= size; ++i) {
        for (int j = 1; j <= i; ++j) {
            CHECK(clSetKernelArg(krn, 0, sizeof(int), &i),    "SetArg0");
            CHECK(clSetKernelArg(krn, 1, sizeof(int), &j),    "SetArg1");
            CHECK(clSetKernelArg(krn, 2, sizeof(cl_mem), &d_arr),"SetArg2");
            CHECK(clSetKernelArg(krn, 3, sizeof(int), &size), "SetArg3");

            cl_int launch_err = clEnqueueNDRangeKernel(
                cq, krn, 1, NULL, &gws, &lws, 0, NULL, &ev);
        }
    }

    err = clEnqueueReadBuffer(
        cq, d_arr, CL_TRUE, 0,
        size * sizeof(int),
        arr, 0, NULL, NULL);

    clReleaseMemObject(d_arr);
    clReleaseKernel(krn);
    clReleaseProgram(prog);
    clReleaseCommandQueue(cq);
    clReleaseContext(ctx);

    return 0;
}
