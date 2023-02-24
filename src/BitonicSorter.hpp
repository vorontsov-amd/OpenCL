#pragma once
#include <cmath>
#include <fstream>



#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>


namespace OpenCLApp {

    enum SortDirection {
        INCREASING = 0,
        DECREASING = -1
    };

    template <typename T>
    class BitonicSorter final
    {
    private:
        cl::Platform           platform_;
        cl::vector<cl::Device> devices_;
        cl::Context            context_;
        cl::Program            program_;
        cl::vector<cl::Kernel> kernels_;
        cl::CommandQueue       queue_;
    public:
        BitonicSorter();
        
        template <typename Iterator>
        void operator() (Iterator begin, Iterator end, SortDirection direction = INCREASING); 

    private:
        cl::vector<cl::Device> initDevices();
        cl::Program initProgram();
        cl::vector<cl::Kernel> initKernels();
    };

};





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

    template <typename T>
    cl::vector<cl::Device> BitonicSorter<T>::initDevices() {
        
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

    template <typename T>
    cl::Program BitonicSorter<T>::initProgram() {
        
        std::ifstream programFile {SOURCE_FILE_NAME};
        std::string programString(std::istreambuf_iterator<char>(programFile),
                                 (std::istreambuf_iterator<char>()));

        cl::Program::Sources source {programString};
        
        cl::Program program {context_, source};
        program_.build(devices_);

        return program;
    }

    template <typename T>
    cl::vector<cl::Kernel> BitonicSorter<T>::initKernels() {
        cl::vector<cl::Kernel> kernels(5);

        kernels[BSORT_INIT]       = {program_, "bsort_init"};
        kernels[BSORT_STAGE_0]    = {program_, "bsort_stage_0"};
        kernels[BSORT_STAGE_N]    = {program_, "bsort_stage_n"};
        kernels[BSORT_MERGE]      = {program_, "bsort_merge"};
        kernels[BSORT_MERGE_LAST] = {program_, "bsort_merge_last"};

        return kernels;
    }

    template <typename T>
    BitonicSorter<T>::BitonicSorter() : 
        platform_ {cl::Platform::get()},
        devices_  {initDevices()},
        context_  {devices_},
        program_  {initProgram()},
        kernels_  {initKernels()},
        queue_    {context_, devices_[0]}
        {}

    template <typename T>
    template <typename Iterator> 
    void BitonicSorter<T>::operator() (Iterator begin, Iterator end, SortDirection direction) {
        using Type = typename std::iterator_traits<Iterator>::value_type;

        auto numOfElem  = std::distance(begin, end) + 1;
        size_t capacity = 1 << (int)std::ceil(log2(numOfElem));

        Type aggregate = std::numeric_limits<Type>::max();
        if (direction == DECREASING) aggregate = -aggregate;

        std::vector<Type> data(capacity, aggregate);
        std::copy(begin, end, data.begin());

        cl::Buffer buffer(context_, data.begin(), data.end(), false);

        auto local_size = kernels_[BSORT_INIT].getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(devices_[0]);
        size_t global_size = capacity / 8;

        local_size = 1 << (int)trunc(log2(local_size)); 

        if(global_size < local_size) {
            local_size = global_size;
        }

        kernels_[BSORT_INIT].setArg(0, buffer);
        kernels_[BSORT_INIT].setArg(1, 8 * local_size * sizeof(Type), NULL);

        // Create queue and enqueue kernel-execution command
        cl::NDRange offset {0};
        cl::NDRange range {1};
        queue_.enqueueNDRangeKernel(kernels_[BSORT_INIT], offset, global_size, local_size);

        kernels_[BSORT_STAGE_0].setArg(0, buffer);
        kernels_[BSORT_STAGE_0].setArg(1, 8 * local_size * sizeof(Type), NULL);
        kernels_[BSORT_STAGE_N].setArg(0, buffer);
        kernels_[BSORT_STAGE_N].setArg(1, 8 * local_size * sizeof(Type), NULL);

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
        kernels_[BSORT_MERGE].setArg(1, 8 * local_size * sizeof(Type), NULL);
        kernels_[BSORT_MERGE_LAST].setArg(0, buffer);
        kernels_[BSORT_MERGE_LAST].setArg(1, 8 * local_size * sizeof(Type), NULL);

        /* Set the sort direction */
        kernels_[BSORT_MERGE].setArg(3, direction);
        kernels_[BSORT_MERGE_LAST].setArg(2, direction);

        /* Perform the bitonic merge */
        for(int stage = num_stages; stage > 1; stage >>= 1) {
            kernels_[BSORT_MERGE].setArg(2, stage);
            queue_.enqueueNDRangeKernel(kernels_[BSORT_MERGE], offset, global_size, local_size); 
        }

        queue_.enqueueNDRangeKernel(kernels_[BSORT_MERGE_LAST], offset, global_size, local_size); 

        cl::copy(queue_, buffer, begin, end);
    }

};




