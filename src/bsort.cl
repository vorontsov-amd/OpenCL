#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// #define TYPE double4
// #define COMPORATOR_TYPE long4
// #define MASK_TYPE ulong4
// #define TYPE_CAST as_ulong4

#define UP 0
#define DOWN -1

#define mask_1 (MASK_TYPE)(1, 0, 3, 2)
#define mask_2 (MASK_TYPE)(2, 3, 0, 1)
#define mask_3 (MASK_TYPE)(3, 2, 1, 0)

#define add_mask_1 (COMPORATOR_TYPE)(1, 1, 3, 3)
#define add_mask_2 (COMPORATOR_TYPE)(2, 3, 2, 3)
#define add_mask_3 (COMPORATOR_TYPE)(1, 2, 2, 3)
#define add_mask_4 (COMPORATOR_TYPE)(4, 5, 6, 7)


/* Sort elements in a vector */
#define SORT_VECTOR(input, dir)                                    \
   comp = (input < shuffle(input, mask_1)) ^ dir;                  \
   input = shuffle(input, TYPE_CAST(comp     + add_mask_1));       \
   comp = (input < shuffle(input, mask_2)) ^ dir;                  \
   input = shuffle(input, TYPE_CAST(comp * 2 + add_mask_2));       \
   comp = (input < shuffle(input, mask_3)) ^ dir;                  \
   input = shuffle(input, TYPE_CAST(comp     + add_mask_3));       \

/* Sort elements between two vectors */
#define SWAP_VECTORS(input1, input2, dir)                         \
   temp = input1;                                                 \
   comp = ((input1 < input2) ^ dir) * 4 + add_mask_4;             \
   input1 = shuffle2(input1, input2, TYPE_CAST(comp));            \
   input2 = shuffle2(input2, temp,   TYPE_CAST(comp));            \



/* Perform initial sort */
__kernel void bsort_init(__global TYPE *g_data, __local TYPE *l_data) {

   TYPE temp;
   COMPORATOR_TYPE comp;

   uint id = get_local_id(0) * 2;
   uint global_start = get_group_id(0) * get_local_size(0) * 2 + id;

   TYPE input1 = g_data[global_start]; 
   TYPE input2 = g_data[global_start + 1];

   /* Sort input 1 - ascending, input 2 - descending */
   SORT_VECTOR(input1, UP);
   SORT_VECTOR(input2, DOWN);    

   /* Swap corresponding elements of input 1 and 2 */
   int dir = (get_local_id(0) % 2) * -1;
   SWAP_VECTORS(input1, input2, dir); 

   /* Sort data and store in local memory */
   SORT_VECTOR(input1, dir);
   SORT_VECTOR(input2, dir);
   l_data[id] = input1;
   l_data[id+1] = input2;

   /* Create bitonic set */
   for(uint size = 2; size < get_local_size(0); size <<= 1) {
      dir = (get_local_id(0)/size & 1) * -1;

      for(uint stride = size; stride > 1; stride >>= 1) {
         barrier(CLK_LOCAL_MEM_FENCE);
         id = get_local_id(0) + (get_local_id(0)/stride)*stride;
         SWAP_VECTORS(l_data[id], l_data[id + stride], dir)
      }

      barrier(CLK_LOCAL_MEM_FENCE);
      id = get_local_id(0) * 2;
      input1 = l_data[id]; input2 = l_data[id+1];
      SWAP_VECTORS(input1, input2, dir);
      SORT_VECTOR(input1, dir);
      SORT_VECTOR(input2, dir);
      l_data[id] = input1;
      l_data[id+1] = input2;
   }

   /* Perform bitonic merge */
   dir = (get_group_id(0) % 2) * -1;
   for(uint stride = get_local_size(0); stride > 1; stride >>= 1) {
      barrier(CLK_LOCAL_MEM_FENCE);
      id = get_local_id(0) + (get_local_id(0)/stride)*stride;
      SWAP_VECTORS(l_data[id], l_data[id + stride], dir)
   }
   barrier(CLK_LOCAL_MEM_FENCE);

   /* Perform final sort */
   id = get_local_id(0) * 2;
   input1 = l_data[id]; input2 = l_data[id+1];
   SWAP_VECTORS(input1, input2, dir);
   SORT_VECTOR(input1, dir);
   SORT_VECTOR(input2, dir);
   g_data[global_start] = input1;
   g_data[global_start+1] = input2;

}


/* Sort the bitonic set */
__kernel void bsort_merge(__global TYPE *g_data, __local TYPE *l_data, uint stage, int dir) {
   
   TYPE temp;
   COMPORATOR_TYPE comp;

   /* Determine location of data in global memory */
   uint global_start  = (get_group_id(0) + (get_group_id(0)/stage)*stage) * 
      get_local_size(0) + get_local_id(0);
   uint global_offset = stage * get_local_size(0);

   /* Perform swap */
   TYPE input1 = g_data[global_start];
   TYPE input2 = g_data[global_start + global_offset];

   SWAP_VECTORS(input1, input2, dir); 
   g_data[global_start] = input1;
   g_data[global_start + global_offset] = input2;
}




/* Perform final step of the bitonic merge */
__kernel void bsort_merge_last(__global TYPE *g_data, __local TYPE *l_data, int dir) {

   TYPE temp;
   COMPORATOR_TYPE comp;

   /* Determine location of data in global memory */
   uint id = get_local_id(0);
   uint global_start = get_group_id(0) * get_local_size(0) * 2 + id;

   /* Perform initial swap */
   TYPE input1 = g_data[global_start];
   TYPE input2 = g_data[global_start + get_local_size(0)];

   SWAP_VECTORS(input1, input2, dir);
   l_data[id] = input1;
   l_data[id + get_local_size(0)] = input2;

   /* Perform bitonic merge */
   for(uint stride = get_local_size(0)/2; stride > 1; stride >>= 1) {
      barrier(CLK_LOCAL_MEM_FENCE);
      id = get_local_id(0) + (get_local_id(0)/stride)*stride;
      SWAP_VECTORS(l_data[id], l_data[id + stride], dir)
   }
   barrier(CLK_LOCAL_MEM_FENCE);

   /* Perform final sort */
   id = get_local_id(0) * 2;
   input1 = l_data[id]; input2 = l_data[id+1];
   SWAP_VECTORS(input1, input2, dir);
   SORT_VECTOR(input1, dir);
   SORT_VECTOR(input2, dir);

   /* Store the result to global memory */
   g_data[global_start + get_local_id(0)] = input1;
   g_data[global_start + get_local_id(0) + 1] = input2;
}


/* Perform successive stages of the bitonic sort */
__kernel void bsort_first_stage(__global TYPE *g_data, __local TYPE *l_data, uint stage, uint high_stage) {
   
   int dir = (get_group_id(0)/high_stage & 1) * -1;
   bsort_merge(g_data, l_data, stage, dir);
}


/* Perform lowest stage of the bitonic sort */
__kernel void bsort_second_stage(__global TYPE *g_data, __local TYPE *l_data, uint high_stage) {

   int dir = (get_group_id(0)/high_stage & 1) * -1;
   bsort_merge_last(g_data, l_data, dir);
}



