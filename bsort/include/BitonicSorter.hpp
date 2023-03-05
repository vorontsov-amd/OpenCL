#pragma once
#include <cmath>
#include <fstream>
#include <climits>
#include <iostream>
#include "sourcePath.h"
#include <bit>
#include <cassert>

#define CL_HPP_TARGET_OPENCL_VERSION 220
#define CL_HPP_ENABLE_EXCEPTIONS

#ifdef MAC
    #include <OpenCL/cl.hpp>
#else
    #include <CL/opencl.hpp>
#endif


namespace OpenCLApp {

    enum SortDirection {
        INCREASING = 0,
        DECREASING = -1
    };

    enum class Platform {
        NVIDIA, INTEL, ANY_PLATFORM
    };

    template <typename T>
    class BitonicSorter final
    {
    private:
        cl::vector<cl::Device> devices_;
        cl::Platform           platform_;
        cl::Context            context_;
        cl::Program            program_;
        cl::CommandQueue       queue_;

        cl::KernelFunctor<cl::Buffer, cl::LocalSpaceArg>                     bsortlInit_;
        cl::KernelFunctor<cl::Buffer, cl::LocalSpaceArg, unsigned, unsigned> bsortFirstStage_;
        cl::KernelFunctor<cl::Buffer, cl::LocalSpaceArg, unsigned>           bsortSecondStage_;
        cl::KernelFunctor<cl::Buffer, cl::LocalSpaceArg, unsigned, int>      bsortMerge_;
        cl::KernelFunctor<cl::Buffer, cl::LocalSpaceArg, int>                bsortMergeLast_;

    public:
        BitonicSorter(Platform requiredPlatform = Platform::ANY_PLATFORM);
        
        template <typename Iterator>
        void operator() (Iterator begin, Iterator end, SortDirection direction = INCREASING); 
        std::string getOpenCLAppInfo(cl::Error& err) noexcept;

    private:
        cl::Platform initPlatform(Platform requiredPlatform);
        cl::Program initProgram();

        template <typename KernelFunctor> 
        cl::size_type localSize(KernelFunctor&& functor, size_t global_ize);

        cl::Platform FindPlatform(const cl::vector<cl::Platform>& platforms, std::string platform_name);
        void InitDevices(const cl::Platform& platform, cl::vector<cl::Device>& devices);
        bool CheckDevices(const cl::Platform& platform);

    };
};




namespace OpenCLApp {

    namespace {
        const char* SOURCE_FILE_NAME = CL_PROGRAM_PATH;

        std::string toString(cl_bool x) {
            if (x) return "true";
            return "false";
        }

        template <typename Iterator>
        size_t getBufferCapacity(Iterator begin, Iterator end) {
            size_t numOfElem  = std::distance(begin, end) + 1;
            size_t capacity = 1 << (CHAR_BIT * sizeof(numOfElem) - (std::countl_zero(numOfElem)));
            if (capacity < 16) capacity = 16;
            return capacity;
        }

    };

    //------------------------------------------------------------------------------------------------------------------------------

    template <typename T>
    cl::Platform BitonicSorter<T>::FindPlatform(const cl::vector<cl::Platform>& platforms, std::string platform_name) {
        auto platform_it = std::find_if(platforms.begin(), platforms.end(), [&](cl::Platform platform) {
            auto pl_name = platform.getInfo<CL_PLATFORM_NAME>();
            return pl_name.find(platform_name) != pl_name.size();
        });

        if (platform_it != platforms.end()) {
            InitDevices(*platform_it, devices_);
            return *platform_it;
        }
        else throw std::runtime_error("Can't find platform " + platform_name);
    }

    //------------------------------------------------------------------------------------------------------------------------------

    template <typename T>
    bool BitonicSorter<T>::CheckDevices(const cl::Platform& platform) {
        cl::vector<cl::Device> device;
        try {
            InitDevices(platform, device);
            return true;
        }
        catch (cl::Error& err) {
            return false;
        }
    }

    //------------------------------------------------------------------------------------------------------------------------------

    template <typename T>
    void BitonicSorter<T>::InitDevices(const cl::Platform& platform, cl::vector<cl::Device>& devices) {
        try {
            platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        } 
        catch (cl::Error& error) {
            platform.getDevices(CL_DEVICE_TYPE_CPU, &devices);
        }
    }

    //------------------------------------------------------------------------------------------------------------------------------

    template <typename T>
    cl::Platform BitonicSorter<T>::initPlatform(Platform requiredPlatform) {
        
        cl::vector<cl::Platform> platforms;

        cl::Platform::get(&platforms);

        if (requiredPlatform == Platform::NVIDIA) {
            return FindPlatform(platforms, "NVIDIA");
        } else if (requiredPlatform == Platform::INTEL) {
            return FindPlatform(platforms, "Inter(R)");
        } else {
            for (auto&& platform : platforms) {    
                if (CheckDevices(platform)) {
                    InitDevices(platform, devices_);
                    return platform;
                }
            }
        }

        throw std::runtime_error("Can't find any platform");
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
    BitonicSorter<T>::BitonicSorter(Platform requiredPlatform) try : 
        platform_         {initPlatform(requiredPlatform)},
        context_          {devices_},
        program_          {initProgram()},
        queue_            {context_, devices_[0]},
        bsortlInit_       {program_, "bsort_init"},
        bsortFirstStage_  {program_, "bsort_first_stage"},
        bsortSecondStage_ {program_, "bsort_second_stage"},
        bsortMerge_       {program_, "bsort_merge"},
        bsortMergeLast_   {program_, "bsort_merge_last"}
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
    template <typename KernelFunctor> 
    cl::size_type BitonicSorter<T>::localSize(KernelFunctor&& functor, size_t global_size) {
        auto local_size = functor.getKernel().template getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(devices_[0]);
        local_size = 1 << (CHAR_BIT * sizeof(local_size) - std::countl_zero(local_size) - 1); 
        if(global_size < local_size) {
            local_size = global_size;
        }
        return local_size;
    }

    //------------------------------------------------------------------------------------------------------------------------------

    template <typename T>
    template <typename Iterator> 
    void BitonicSorter<T>::operator() (Iterator begin, Iterator end, SortDirection direction) {

        /* Create buffer */
        size_t capacity = getBufferCapacity(begin, end);

        T aggregate = std::numeric_limits<T>::max();
        if (direction == DECREASING) aggregate = std::numeric_limits<T>::lowest();

        std::vector<T> data(capacity, aggregate);
        std::copy(begin, end, data.begin());
        
        cl::Buffer buffer(context_, data.begin(), data.end(), false);

        /* Determine maximum work-group size */
        size_t global_size = capacity / 8;
        auto local_size = localSize(bsortlInit_, capacity);

        /* Enqueue initial sorting kernel */
        cl::EnqueueArgs args {queue_, cl::NullRange, global_size, local_size};
        auto localBuffer = cl::Local(8 * local_size * sizeof(T));
        bsortlInit_(args, buffer, localBuffer);

        /* Execute further stages */
        int num_stages = global_size/local_size;        
        for(int high_stage = 2; high_stage < num_stages; high_stage <<= 1) {
            for(int stage = high_stage; stage > 1; stage >>= 1) {
                bsortFirstStage_(args, buffer, localBuffer, stage, high_stage);
            }
            bsortSecondStage_(args, buffer, localBuffer, high_stage);
        }

        /* Perform the bitonic merge */
        for(int stage = num_stages; stage > 1; stage >>= 1) {
            bsortMerge_(args, buffer, localBuffer, stage, direction); 
        }
        bsortMergeLast_(args, buffer, localBuffer, direction);

        cl::copy(queue_, buffer, begin, end);
    }

    //------------------------------------------------------------------------------------------------------------------------------

};




