
#In tests:

* for_paper -> cross_product (rename), check hip/sycl tests correctness, add hip/sycl tests to cmake
* matmul: check logic of sycl tests (diff checks correct layout), add sycl tests to cmake

#In general

Add queue module. Abstraction around CUDA/HIP streams and sycl queues. Think we need to implement thin class. I.e. lightly copyable class - handler for streams and maybe pointer or shared pointer for sycl. 
RATIONALE: if use thick classes rightaway in some cases when storing references for queues (in for_each for example) we will use pointers or shared pointers which is overhead for streams where we can simply use handler. So presuming lightwightness with emulating it when there is no one (for sycl queues) seems to be prefferable.
Memory classes must take queue as last argument and use queue singleton as default. queue_type::inst() or queue_type::default() - something like that. Later, arrays operations like init also must take queue with default value. For_each also should use queues.

Add cmake INTERFACE.

##Utils
  add init_sycl in some form. ISSUE: what arguments? Cannot use init_cuda (no pci_id, for example), mb string specification like "cpu"/"device"/"any"?
  Add data_traits<> for data general work. data_traits<T>::ptr(T&) will return plain pointer. Optionally may add data_data_traits<T>::have_size trait and traits<T>::size(T&). May be specialized for pointers, thrust iterators, stl and thrust vectors, scfd arrays. May be used for example in reductions new module.
  Add AD from old code. Don't forget to specialize is_floating_point for them. Add scalar_traits.

##Reductions
  Disaster for now. Use data_traits. What to do with operation specification? Add cub version.

##Sycl 
  Try idea with is_device_copyable.

##Arrays + Sycl
  
  Anyway sycl correct layout depends on runtime (cpu/gpu device) so we need to add strided arranger that
  can use runtime information to switch layout. Also strided arranger can be used to create slices and subarrays of different kinds.
  In Memory we cannot always have is_array_of_structs_prefferable as static const member. Perhaps use it when 
  possible and use is_array_of_structs_prefferable_runtime(queue) static method when not. Outer algorithms must 
  choose behaviour using template programming.
  Also we presumably will have second field like this - is_last_index_fast_prefferable - specially for SYCL situation when nd arrays are cpu-like but tensor part is device dependent. Runtime version is also needed.
  Default arranger perhaps will use ND knoledge. Have to thinkthrough - how and when exactly prefferable layout will be passed to the arranger (in arrays constructor?). Maybe have to add special sycl arranger based of strided general arranger. 

##Arrays 

  shared_array, uniqe_array. Slices and subarrays.
  Some sort of Initializer. ISSUE problems with kernels from pure CPP code. Want to leave ability to use arrays from pure CPP code like thrust::device_vector. Zero-initialization with default zero initializable types? Static check for non-trivially constructable types? ValueInitializer? Perhaps need to add fill_zero to Memory.

##For_each

  * 1-2-3 for_each_nd for cuda hip sycl must be implemeneted using nd kernels native way. Block size if any must be specified in nd-form.
  * Try to optimize openmp version - seems large performance break (comparing for example with sycl cpu version), especially for small sizes. Try to use strides?
  * TBB for_each.

##Communication

  * distributor must be templated on communicator type. 
  * Add isend/irecv/send/recv to mpi communicator. More operations?
  * Add serial communicator with all mock operations. (isend/irecv are not mock in this case).
  * add angle data transfers ability to distributor.
  * add general elements distributor (adapt from old code).
