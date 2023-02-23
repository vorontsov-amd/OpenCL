#define CL_HPP_ENABLE_EXCEPTIONS

#include <fstream>
#include <iterator>
#include <iostream>
#include <CL/opencl.hpp>
#include <iostream>
#include <random>

#define SZ (1 << 20)

int main(void) {
   
   cl::vector<cl::Platform> platforms;
   cl::vector<cl::Device> devices;

   std::random_device rd;
   std::mt19937 gen(rd());
   std::uniform_int_distribution<> random(0, 50000);
   float data[SZ] {};
   for (auto& x: data) {
      x = random(gen);
   }

   try {
      // Place the GPU devices of the first platform into a context
      cl::Platform::get(&platforms);
      platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
      cl::Context context(devices);
      
      // Create kernel
      std::ifstream programFile("bsort.cl");
      std::string programString(std::istreambuf_iterator<char>(programFile),
                               (std::istreambuf_iterator<char>()));

      cl::Program::Sources source {programString};
      cl::Program program(context, source);
      program.build(devices);
      cl::Kernel kernel_init(program, "bsort_init");

      // Create buffer and make it a kernel argument
      cl::Buffer buffer(context, 
            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(data), data);


      auto local_size = kernel_init.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(devices[0]);
      size_t global_size = SZ/8;

      local_size = (int)pow(2, trunc(log2(local_size))); 

      if(global_size < local_size) {
         local_size = global_size;
      }
      // local_size = global_size / 8;

      std::cout << "local size = " << local_size << ", global size = " << global_size << '\n';

      kernel_init.setArg(0, buffer);
      kernel_init.setArg(1, 8 * local_size * sizeof(float), NULL);

      // Create queue and enqueue kernel-execution command
      cl::CommandQueue queue(context, devices[0]);
      cl::NDRange offset {0};
      cl::NDRange range {1};
      queue.enqueueNDRangeKernel(kernel_init, offset, global_size, local_size);
      // queue.finish();
      // queue.enqueueReadBuffer(buffer, CL_TRUE, 0, sizeof(data), data);

      // for (auto x: data) {
      //    std::cout << x << ' ';
      // }
      // std::cout << "\n\n";

      cl::Kernel kernel_stage_0{program, "bsort_stage_0"};
      cl::Kernel kernel_stage_n{program, "bsort_stage_n"};

      kernel_stage_0.setArg(0, buffer);
      kernel_stage_0.setArg(1, 8 * local_size * sizeof(float), NULL);
      kernel_stage_n.setArg(0, buffer);
      kernel_stage_n.setArg(1, 8 * local_size * sizeof(float), NULL);


      int num_stages = global_size/local_size;
      for(int high_stage = 2; high_stage < num_stages; high_stage <<= 1) {

         kernel_stage_0.setArg(2, high_stage);      
         kernel_stage_n.setArg(3, high_stage);

         for(int stage = high_stage; stage > 1; stage >>= 1) {

            kernel_stage_n.setArg(2, stage);
            queue.enqueueNDRangeKernel(kernel_stage_n, offset, global_size, local_size); 
         }

         queue.enqueueNDRangeKernel(kernel_stage_0, offset, global_size, local_size); 
      }

      // queue.enqueueReadBuffer(buffer, CL_TRUE, 0, sizeof(data), data);

      // for (auto x: data) {
      //    std::cout << x << ' ';
      // }
      // std::cout << "\n\n";


      cl::Kernel kernel_merge {program, "bsort_merge"};
      cl::Kernel kernel_merge_last {program, "bsort_merge_last"};

      kernel_merge.setArg(0, buffer);
      kernel_merge.setArg(1, 8 * local_size * sizeof(float), NULL);
      kernel_merge_last.setArg(0, buffer);
      kernel_merge_last.setArg(1, 8 * local_size * sizeof(float), NULL);

      /* Set the sort direction */
      int direction = 0;
      kernel_merge.setArg(3, direction);
      kernel_merge_last.setArg(2, direction);

      /* Perform the bitonic merge */
      for(int stage = num_stages; stage > 1; stage >>= 1) {

         kernel_merge.setArg(2, stage);

         queue.enqueueNDRangeKernel(kernel_merge, offset, global_size, local_size); 
      }

      queue.enqueueNDRangeKernel(kernel_merge_last, offset, global_size, local_size); 

      /* Read the result */
      queue.enqueueReadBuffer(buffer, CL_TRUE, 0, sizeof(data), data);



      // for (auto x: data) {
      //    std::cout << x << ' ';
      // }
      // std::cout << "\n\n";

   }
   catch(cl::Error& e) {
      std::cout << e.what() << ": Error code " << e.err() << std::endl;   
   }

   return 0;
}
