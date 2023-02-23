#pragma once

#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>


namespace OpenCLApp {

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
        void operator() (Iterator begin, Iterator end); 

    private:
        cl::vector<cl::Device> initDevices();
        cl::Program initProgram();
        cl::vector<cl::Kernel> initKernels();
    };

};




