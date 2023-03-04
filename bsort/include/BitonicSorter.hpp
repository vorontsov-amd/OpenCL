#pragma once
#include <cmath>
#include <fstream>
#include <iostream>
#include "sourcePath.h"
#include <new>

#define CL_HPP_TARGET_OPENCL_VERSION 220
#define CL_HPP_ENABLE_EXCEPTIONS

#ifdef MAC
    #include <OpenCL/cl.h>
#else
    #include <CL/opencl.hpp>
#endif

namespace OpenCLApp {

    enum SortDirection {
        INCREASING = 0,
        DECREASING = -1
    };

    template <typename T>
    class BitonicSorter final
    {
    private:
        cl::vector<cl::Device> devices_;
        cl::Platform           platform_;
        cl::Context            context_;
        cl::Program            program_;
        cl::vector<cl::Kernel> kernels_;
        cl::CommandQueue       queue_;
    public:
        BitonicSorter();
        
        template <typename Iterator>
        void operator() (Iterator begin, Iterator end, SortDirection direction = INCREASING); 
        std::string getOpenCLAppInfo(cl::Error& err) noexcept;

    private:
        cl::Platform initPlatform();
        cl::vector<cl::Device> initDevices();
        cl::Program initProgram();
        cl::vector<cl::Kernel> initKernels();        
    };

};




namespace OpenCLApp {

    namespace {
        const char* SOURCE_FILE_NAME = CL_PROGRAM_PATH;

        enum KernelID {
            BSORT_INIT,
            BSORT_FIRST_STAGE,
            BSORT_SECOND_STAGE,
            BSORT_MERGE,
            BSORT_MERGE_LAST
        };

        std::string toString(cl_bool x) {
            if (x) return "true";
            return "false";
        }

    };

    //------------------------------------------------------------------------------------------------------------------------------

    template <typename T>
    cl::Platform BitonicSorter<T>::initPlatform() {
        
        cl::vector<cl::Platform> platforms;

        cl::Platform::get(&platforms);
        for (auto&& platform : platforms) {
            try {
                platform.getDevices(CL_DEVICE_TYPE_GPU, &devices_);
            } 
            catch (cl::Error& error) {
                platform.getDevices(CL_DEVICE_TYPE_CPU, &devices_);
            }
            if (!devices_.empty()) return platform;
        }

        throw std::runtime_error("Can't find any platform");
    }


    //------------------------------------------------------------------------------------------------------------------------------

    template <typename T>
    cl::vector<cl::Device> BitonicSorter<T>::initDevices() {
        
        cl::vector<cl::Device> devices;

        try {
            platform_.getDevices(CL_DEVICE_TYPE_GPU, &devices);
            return devices;
        }

        catch (cl::Error& error) {
            if (error.err() == CL_DEVICE_NOT_FOUND) {
                platform_.getDevices(CL_DEVICE_TYPE_CPU, &devices);
                return devices;
            }
            else throw; 
        }
    }

    //------------------------------------------------------------------------------------------------------------------------------

    template <>
    cl::Program BitonicSorter<float>::initProgram() {

        std::ifstream programFile {SOURCE_FILE_NAME};
        std::string programString(std::istreambuf_iterator<char>(programFile),
                                 (std::istreambuf_iterator<char>()));

        cl::Program::Sources source {programString};
        
        cl::Program program {context_, source};
        
        const char type[] = "-DTYPE=float4 -DCOMPORATOR_TYPE=int4 -DMASK_TYPE=uint4 -DTYPE_CAST=as_uint4";
        program.build(devices_, type);

        return program;
    }
    
    //------------------------------------------------------------------------------------------------------------------------------

    template <>
    cl::Program BitonicSorter<int>::initProgram() {
        
        std::ifstream programFile {SOURCE_FILE_NAME};
        std::string programString(std::istreambuf_iterator<char>(programFile),
                                 (std::istreambuf_iterator<char>()));

        cl::Program::Sources source {programString};
        
        cl::Program program {context_, source};
        
        const char type[] = "-DTYPE=int4 -DCOMPORATOR_TYPE=int4 -DMASK_TYPE=uint4 -DTYPE_CAST=as_uint4";
        program.build(devices_, type);
        return program;
    }

    //------------------------------------------------------------------------------------------------------------------------------

    template <>
    cl::Program BitonicSorter<unsigned>::initProgram() {
        
        std::ifstream programFile {SOURCE_FILE_NAME};
        std::string programString(std::istreambuf_iterator<char>(programFile),
                                 (std::istreambuf_iterator<char>()));

        cl::Program::Sources source {programString};
        
        cl::Program program {context_, source};
        
        const char type[] = "-DTYPE=uint4 -DCOMPORATOR_TYPE=int4 -DMASK_TYPE=uint4 -DTYPE_CAST=as_uint4";
        program.build(devices_, type);
        return program;
    }

    //------------------------------------------------------------------------------------------------------------------------------

    template <>
    cl::Program BitonicSorter<double>::initProgram() {
        
        std::ifstream programFile {SOURCE_FILE_NAME};
        std::string programString(std::istreambuf_iterator<char>(programFile),
                                 (std::istreambuf_iterator<char>()));

        cl::Program::Sources source {programString};
        
        cl::Program program {context_, source};
        
        const char type[] = "-DTYPE=double4 -DCOMPORATOR_TYPE=long4 -DMASK_TYPE=ulong4 -DTYPE_CAST=as_ulong4";
        program.build(devices_, type);
        return program;
    }

    //------------------------------------------------------------------------------------------------------------------------------

    template <>
    cl::Program BitonicSorter<int64_t>::initProgram() {
        
        std::ifstream programFile {SOURCE_FILE_NAME};
        std::string programString(std::istreambuf_iterator<char>(programFile),
                                 (std::istreambuf_iterator<char>()));

        cl::Program::Sources source {programString};
        
        cl::Program program {context_, source};
        
        const char type[] = "-DTYPE=long4 -DCOMPORATOR_TYPE=long4 -DMASK_TYPE=ulong4 -DTYPE_CAST=as_ulong4";
        program.build(devices_, type);
        return program;
    }

    //------------------------------------------------------------------------------------------------------------------------------

    template <>
    cl::Program BitonicSorter<uint64_t>::initProgram() {
        
        std::ifstream programFile {SOURCE_FILE_NAME};
        std::string programString(std::istreambuf_iterator<char>(programFile),
                                 (std::istreambuf_iterator<char>()));

        cl::Program::Sources source {programString};
        
        cl::Program program {context_, source};
        
        const char type[] = "-DTYPE=ulong4 -DCOMPORATOR_TYPE=long4 -DMASK_TYPE=ulong4 -DTYPE_CAST=as_ulong4";
        program.build(devices_, type);
        return program;
    }

    //------------------------------------------------------------------------------------------------------------------------------

    template <typename T>
    cl::vector<cl::Kernel> BitonicSorter<T>::initKernels() {
        cl::vector<cl::Kernel> kernels(5);

        kernels[BSORT_INIT]        = {program_, "bsort_init"};
        kernels[BSORT_FIRST_STAGE] = {program_, "bsort_first_stage"};
        kernels[BSORT_SECOND_STAGE]= {program_, "bsort_second_stage"};
        kernels[BSORT_MERGE]       = {program_, "bsort_merge"};
        kernels[BSORT_MERGE_LAST]  = {program_, "bsort_merge_last"};

        return kernels;
    }

    //------------------------------------------------------------------------------------------------------------------------------

    template <typename T>
    BitonicSorter<T>::BitonicSorter() try : 
        platform_ {initPlatform()},
        context_  {devices_},
        program_  {initProgram()},
        kernels_  {initKernels()},
        queue_    {context_, devices_[0]}
        {}

    catch (cl::Error& error) {
        if (error.err() == CL_BUILD_PROGRAM_FAILURE) {
            for (auto& dev: devices_) {
                auto status = program_.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(dev);
                if (status != CL_BUILD_ERROR)
                    continue;

                auto name = dev.getInfo<CL_DEVICE_NAME>();
                auto buildlog = program_.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev);
                std::cerr << "Build log for " << name << ":" << std::endl
                          << buildlog << std::endl;
            }
        }
        else throw std::runtime_error(getOpenCLAppInfo(error));
    }
    
    //------------------------------------------------------------------------------------------------------------------------------

    template <typename T> 
    std::string BitonicSorter<T>::getOpenCLAppInfo(cl::Error& err) noexcept {

        std::string log;
        log += "current Error: ";
        log += err.what();
        log += ". Err code = ";
        log += std::to_string(err.err());
        log += "\n";

        try {
            if (err.err() != CL_PLATFORM_NOT_FOUND_KHR) {
                //Info about Platform-------------------------------------------------
                log += "Platform info:\n";
                log += "CL_PLATFORM_NAME: ";
                log += platform_.getInfo<CL_PLATFORM_NAME>();
                log += "\n";

                log += "CL_PLATFORM_VENDOR: ";
                log += platform_.getInfo<CL_PLATFORM_VENDOR>();
                log += "\n";
                
                log += "CL_PLATFORM_VERSION: ";
                log += platform_.getInfo<CL_PLATFORM_VERSION>();
                log += "\n";

                log += "CL_PLATFORM_PROFILE: ";
                log += platform_.getInfo<CL_PLATFORM_PROFILE>();
                log += "\n";
                //--------------------------------------------------------------------
                //Info about devices--------------------------------------------------
                log += "device size = ";
                log += std::to_string(devices_.size());
                log += "\n";
                for (auto&& device: devices_) {
                    log += "Device info:\n";
                    log += "CL_DEVICE_NAME: ";
                    log += device.getInfo<CL_DEVICE_NAME>();
                    log += "\n";

                    log += "CL_DEVICE_AVAILABLE: ";
                    log += toString(device.getInfo<CL_DEVICE_AVAILABLE>());
                    log += "\n";

                    log += "CL_DEVICE_VERSION: ";
                    log += device.getInfo<CL_DEVICE_VERSION>();
                    log += "\n";

                    log += "CL_DEVICE_COMPILER_AVAILABLE: ";
                    log += toString(device.getInfo<CL_DEVICE_COMPILER_AVAILABLE>());
                    log += "\n";
                }
                //--------------------------------------------------------------------
                //Info about context--------------------------------------------------
                log += "Context info:\n";
                log += "CL_CONTEXT_REFERENCE_COUNT: ";
                log += std::to_string(context_.getInfo<CL_CONTEXT_REFERENCE_COUNT>());
                log += "\n";
                
                log += "CL_CONTEXT_NUM_DEVICES: ";
                log += std::to_string(context_.getInfo<CL_CONTEXT_NUM_DEVICES>());
                log += "\n";
                //--------------------------------------------------------------------
            } else {
                log += "CL_PLATFORM_NOT_FOUND\n";
            }
        }
        catch (cl::Error& error) {
            log += "\nError:\n";
            auto err = error.err();
            //Platform err---------------------------------------------------------
            if (err == CL_INVALID_PLATFORM)   log += "CL_INVALID_PLATFORM\n";
            //Device err-----------------------------------------------------------
            if (err == CL_INVALID_DEVICE)     log += "CL_INVALID_DEVICE\n";
            //Context err----------------------------------------------------------
            if (err == CL_INVALID_CONTEXT)    log += "CL_INVALID_CONTEXT\n";
            //Other err------------------------------------------------------------
            if (err == CL_INVALID_VALUE)      log += "CL_INVALID_VALUE\n";
            if (err == CL_OUT_OF_RESOURCES)   log += "CL_OUT_OF_RESOURCES\n";
            if (err == CL_OUT_OF_HOST_MEMORY) log += "CL_OUT_OF_HOST_MEMORY\n";

            log += "what(): ";
            log += error.what();
            log += ". Err code = ";
            log += std::to_string(error.err());
        }
        catch (std::exception& error) {
            log += "Error in a function that create exception.\n";
            log += error.what();
        }

        return log;
    }

    //------------------------------------------------------------------------------------------------------------------------------

    template <typename T>
    template <typename Iterator> 
    void BitonicSorter<T>::operator() (Iterator begin, Iterator end, SortDirection direction) {

        auto numOfElem  = std::distance(begin, end) + 1;
        size_t capacity = 1 << ((int)std::ceil(log2(numOfElem)) - 1);
        if (capacity < 16) capacity = 16;

        T aggregate = std::numeric_limits<T>::max();
        if (direction == DECREASING) aggregate = std::numeric_limits<T>::lowest();

        std::vector<T> data(capacity, aggregate);
        std::copy(begin, end, data.begin());

        cl::Buffer buffer(context_, data.begin(), data.end(), false);

        auto local_size = kernels_[BSORT_INIT].getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(devices_[0]);
        size_t global_size = capacity / 8;

        local_size = 1 << (int)std::trunc(log2(local_size)); 

        if(global_size < local_size) {
            local_size = global_size;
        }

        kernels_[BSORT_INIT].setArg(0, buffer);
        kernels_[BSORT_INIT].setArg(1, 8 * local_size * sizeof(T), NULL);

        // Create queue and enqueue kernel-execution command
        cl::NDRange offset {0};
        queue_.enqueueNDRangeKernel(kernels_[BSORT_INIT], offset, global_size, local_size);

        kernels_[BSORT_SECOND_STAGE].setArg(0, buffer);
        kernels_[BSORT_SECOND_STAGE].setArg(1, 8 * local_size * sizeof(T), NULL);
        kernels_[BSORT_FIRST_STAGE].setArg(0, buffer);
        kernels_[BSORT_FIRST_STAGE].setArg(1, 8 * local_size * sizeof(T), NULL);

        int num_stages = global_size/local_size;        

        for(int high_stage = 2; high_stage < num_stages; high_stage <<= 1) {

            kernels_[BSORT_SECOND_STAGE].setArg(2, high_stage);      
            kernels_[BSORT_FIRST_STAGE].setArg(3, high_stage);

            for(int stage = high_stage; stage > 1; stage >>= 1) {
                kernels_[BSORT_FIRST_STAGE].setArg(2, stage);
                queue_.enqueueNDRangeKernel(kernels_[BSORT_FIRST_STAGE], offset, global_size, local_size); 
            }

            cl::copy(queue_, buffer, data.begin(), data.end());

            queue_.enqueueNDRangeKernel(kernels_[BSORT_SECOND_STAGE], offset, global_size, local_size); 

            cl::copy(queue_, buffer, data.begin(), data.end());
        }

        kernels_[BSORT_MERGE].setArg(0, buffer);
        kernels_[BSORT_MERGE].setArg(1, 8 * local_size * sizeof(T), NULL);
        kernels_[BSORT_MERGE_LAST].setArg(0, buffer);
        kernels_[BSORT_MERGE_LAST].setArg(1, 8 * local_size * sizeof(T), NULL);

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

    //------------------------------------------------------------------------------------------------------------------------------

};




