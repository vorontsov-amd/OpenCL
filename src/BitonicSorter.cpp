#include "BitonicSorter.hpp"
#include <fstream>
#include <random>
#include <iostream>

namespace OpenCLApp {

    namespace {
        const char* SOURCE_FILE_NAME = "bsort.cl";

        enum KernelID {
            BSORT_INIT,
            BSORT_STAGE_0,
            BSORT_STAGE_N,
            BSORT_MERGE,
            BSORT_MERGE_LAST
        };

    };

    cl::vector<cl::Device> BitonicSorter::initDevices() {
        cl::vector<cl::Device> devices;

        try {
            platform_.getDevices(CL_DEVICE_TYPE_GPU, &devices);
            return devices;
        }

        catch (cl::Error& error) {
            if (error.err() & CL_DEVICE_NOT_FOUND) {
                platform_.getDevices(CL_DEVICE_TYPE_CPU, &devices);
                return devices;
            }
            else throw; 
        }
    }

    cl::Program BitonicSorter::initProgram() {
        
        std::ifstream programFile {SOURCE_FILE_NAME};
        std::string programString(std::istreambuf_iterator<char>(programFile),
                                 (std::istreambuf_iterator<char>()));

        cl::Program::Sources source {programString};
        
        cl::Program program {context_, source};
        program_.build(devices_);

        return program;
    }

    cl::vector<cl::Kernel> BitonicSorter::initKernels() {
        cl::vector<cl::Kernel> kernels(5);

        kernels[BSORT_INIT]       = {program_, "bsort_init"};
        kernels[BSORT_STAGE_0]    = {program_, "bsort_stage_0"};
        kernels[BSORT_STAGE_N]    = {program_, "bsort_stage_n"};
        kernels[BSORT_MERGE]      = {program_, "bsort_merge"};
        kernels[BSORT_MERGE_LAST] = {program_, "bsort_merge_last"};

        return kernels;
    }

    BitonicSorter::BitonicSorter() : 
        platform_ {cl::Platform::get()},
        devices_  {initDevices()},
        context_  {devices_},
        program_  {initProgram()},
        kernels_  {initKernels()},
        queue_    {context_, devices_[0]}
        {}

    template <typename Iterator> 
    void BitonicSorter::operator() (Iterator begin, Iterator end) {
        using Type = typename std::iterator_traits<Iterator>::value_type;

        //!TODO change it
        // std::vector<Type> data;
        // std::copy(begin, end, std::back_inserter(data));

        // auto SZ = std::distance(begin, end);
        // SZ++;

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> random(0, 150);

        auto SZ = 64;

        float data[SZ] {};
        for (auto& x: data) {
            x = random(gen);
        }

        for (auto& x: data) {
            std::cout << x << ' ';
        }
        std::cout << '\n';

        // Create buffer and make it a kernel argument
        cl::Buffer buffer(context_, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(data), data);

        auto local_size = kernels_[BSORT_INIT].getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(devices_[0]);
        size_t global_size = SZ/8;

        local_size = (int)pow(2, trunc(log2(local_size))); 

        if(global_size < local_size) {
            local_size = global_size;
        }

        // std::cout << "local size = " << local_size << ", global size = " << global_size << '\n';

        kernels_[BSORT_INIT].setArg(0, buffer);
        kernels_[BSORT_INIT].setArg(1, 8 * local_size * sizeof(float), NULL);

        // Create queue and enqueue kernel-execution command
        cl::NDRange offset {0};
        cl::NDRange range {1};
        queue_.enqueueNDRangeKernel(kernels_[BSORT_INIT], offset, global_size, local_size);
        queue_.enqueueReadBuffer(buffer, CL_TRUE, 0, sizeof(data), data);

        // for (auto x: data) {
        //     std::cout << x << " ";
        // }
        // std::cout << '\n';


        kernels_[BSORT_STAGE_0].setArg(0, buffer);
        kernels_[BSORT_STAGE_0].setArg(1, 8 * local_size * sizeof(float), NULL);
        kernels_[BSORT_STAGE_N].setArg(0, buffer);
        kernels_[BSORT_STAGE_N].setArg(1, 8 * local_size * sizeof(float), NULL);

        int num_stages = global_size/local_size;
        for(int high_stage = 2; high_stage < num_stages; high_stage <<= 1) {

            kernels_[BSORT_STAGE_0].setArg(2, high_stage);      
            kernels_[BSORT_STAGE_N].setArg(3, high_stage);

            for(int stage = high_stage; stage > 1; stage >>= 1) {
                kernels_[BSORT_STAGE_N].setArg(2, stage);
                queue_.enqueueNDRangeKernel(kernels_[BSORT_STAGE_N], offset, global_size, local_size); 
            }

            queue_.enqueueNDRangeKernel(kernels_[BSORT_STAGE_0], offset, global_size, local_size); 
        }

        kernels_[BSORT_MERGE].setArg(0, buffer);
        kernels_[BSORT_MERGE].setArg(1, 8 * local_size * sizeof(float), NULL);
        kernels_[BSORT_MERGE_LAST].setArg(0, buffer);
        kernels_[BSORT_MERGE_LAST].setArg(1, 8 * local_size * sizeof(float), NULL);

        /* Set the sort direction */
        int direction = 0;
        kernels_[BSORT_MERGE].setArg(3, direction);
        kernels_[BSORT_MERGE_LAST].setArg(2, direction);

        /* Perform the bitonic merge */
        for(int stage = num_stages; stage > 1; stage >>= 1) {

            kernels_[BSORT_MERGE].setArg(2, stage);

            queue_.enqueueNDRangeKernel(kernels_[BSORT_MERGE], offset, global_size, local_size); 
        }

        queue_.enqueueNDRangeKernel(kernels_[BSORT_MERGE_LAST], offset, global_size, local_size); 

        /* Read the result */
        queue_.enqueueReadBuffer(buffer, CL_TRUE, 0, sizeof(data), data);

        for (auto& x: data) {
            std::cout << x << ' ';
        }
        // std::copy(data.begin(), data.end(), begin);
    }

};



int main() {

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> random(0, 15);
    float data[16] {};
    for (auto& x: data) {
        x = random(gen);
    }

    // for (auto& x: data) {
    //     std::cout << x << ' ';
    // }

    std::cout << std::endl;

    OpenCLApp::BitonicSorter sort;

    sort(&data[0], &data[15]);

    // for (auto& x: data) {
    //     std::cout << x << ' ';
    // }


}


