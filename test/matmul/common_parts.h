#ifndef __COMMON_PARTS_H__SYMBOL
#define __COMMON_PARTS_H__SYMBOL

    if(argc != 4)
    {
        std::cout << "Usage: " << argv[0] << " N iters tests" << std::endl;
        std::cout << "where: N is a size of R^{K^2 * N}, iters is number of iterations for better measurements, K is a static parameter." << std::endl;
        std::cout << "       tests = d/h/a for device, host or all." << std::endl;
        return 1;
    }
    std::size_t N = std::atoi(argv[1]);
    std::size_t number_of_iters = std::atoi(argv[2]);
    char tests = argv[3][0];

    std::size_t total_size = K2 * N;

   __COMMON_PARTS_DEVICE_INIT__

    T *u_ptr_host, *v_ptr_host, *mat_mul_ptr_host;
    T *u_ptr_ok_host, *v_ptr_ok_host, *mat_mul_ptr_ok_host;

    T *u_ptr_dev, *v_ptr_dev, *mat_mul_ptr_dev;          //with incorrect GPU layout i.e. CPU layout
    T *u_ptr_ok_dev, *v_ptr_ok_dev, *mat_mul_ptr_ok_dev; //with correct GPU layout
    T *u_ptr_func_dev, *v_ptr_func_dev, *mat_mul_ptr_func_dev; //func with plain ptr

    T *mat_mul_ptr_check, *mat_mul_ptr_ok_check, *mat_mul_ptr_func_check;

    std::random_device rd;
    std::mt19937 engine{ rd() };
    std::uniform_real_distribution<> dist(-100.0, 100.0);


    u_ptr_host                = reinterpret_cast<T*>( std::malloc(sizeof(T)*total_size ) );
    v_ptr_host                = reinterpret_cast<T*>( std::malloc(sizeof(T)*total_size ) );
    u_ptr_ok_host             = reinterpret_cast<T*>( std::malloc(sizeof(T)*total_size ) );
    v_ptr_ok_host             = reinterpret_cast<T*>( std::malloc(sizeof(T)*total_size ) );
    mat_mul_ptr_host          = reinterpret_cast<T*>( std::malloc(sizeof(T)*total_size ) );
    mat_mul_ptr_check         = reinterpret_cast<T*>( std::malloc(sizeof(T)*total_size ) );
    mat_mul_ptr_ok_host       = reinterpret_cast<T*>( std::malloc(sizeof(T)*total_size ) );
    mat_mul_ptr_ok_check      = reinterpret_cast<T*>( std::malloc(sizeof(T)*total_size ) );


    __COMMON_PARTS_SAFE_CALL__( __COMMON_PARTS_DEVICE_MALLOC__( (void**)&u_ptr_ok_dev       , sizeof(T)*total_size ) );
    __COMMON_PARTS_SAFE_CALL__( __COMMON_PARTS_DEVICE_MALLOC__( (void**)&v_ptr_ok_dev       , sizeof(T)*total_size ) );
    __COMMON_PARTS_SAFE_CALL__( __COMMON_PARTS_DEVICE_MALLOC__( (void**)&mat_mul_ptr_ok_dev , sizeof(T)*total_size ) );

    __COMMON_PARTS_SAFE_CALL__( __COMMON_PARTS_DEVICE_MALLOC__( (void**)&u_ptr_dev        , sizeof(T)*total_size ) );
    __COMMON_PARTS_SAFE_CALL__( __COMMON_PARTS_DEVICE_MALLOC__( (void**)&v_ptr_dev        , sizeof(T)*total_size ) );
    __COMMON_PARTS_SAFE_CALL__( __COMMON_PARTS_DEVICE_MALLOC__( (void**)&mat_mul_ptr_dev  , sizeof(T)*total_size ) );

    __COMMON_PARTS_SAFE_CALL__( __COMMON_PARTS_DEVICE_MALLOC__( (void**)&u_ptr_ok_dev       , sizeof(T)*total_size ) );
    __COMMON_PARTS_SAFE_CALL__( __COMMON_PARTS_DEVICE_MALLOC__( (void**)&v_ptr_ok_dev       , sizeof(T)*total_size ) );
    __COMMON_PARTS_SAFE_CALL__( __COMMON_PARTS_DEVICE_MALLOC__( (void**)&mat_mul_ptr_ok_dev , sizeof(T)*total_size ) );

    __COMMON_PARTS_SAFE_CALL__( __COMMON_PARTS_DEVICE_MALLOC__( (void**)&u_ptr_func_dev       , sizeof(T)*total_size ) );
    __COMMON_PARTS_SAFE_CALL__( __COMMON_PARTS_DEVICE_MALLOC__( (void**)&v_ptr_func_dev       , sizeof(T)*total_size ) );
    __COMMON_PARTS_SAFE_CALL__( __COMMON_PARTS_DEVICE_MALLOC__( (void**)&mat_mul_ptr_func_dev , sizeof(T)*total_size ) );

    mat_mul_ptr_func_check    = reinterpret_cast<T*>( std::malloc(sizeof(T)*total_size ) );


    array_device_t u_dev, v_dev, mat_mul_dev;
    array_host_t u_host, v_host, mat_mul_host;

    u_dev.init(N); v_dev.init(N); mat_mul_dev.init(N);
    u_host.init(N); v_host.init(N); mat_mul_host.init(N);
    array_device_view_t u_dev_view(u_dev), v_dev_view(v_dev), mat_mul_dev_view(mat_mul_dev);
    array_host_t u_host_view(u_host), v_host_view(v_host), mat_mul_host_view(mat_mul_host);

    timer_event_host_t host_e1, host_e2;
    timer_event_device_t device_e1, device_e2;
    timer_event_device_t device_int_e1, device_int_e2;

    #pragma omp parallel for
    for(std::size_t n=0u; n<N; ++n)
    for(std::size_t i=0u; i<K; ++i)
    for(std::size_t j=0u; j<K; ++j)
    {
        T u_ = dist(engine);
        T v_ = dist(engine);

        u_ptr_host[IC(n,i,j)] = u_;
        v_ptr_host[IC(n,i,j)] = v_;

        u_dev_view(n,i,j) = u_;
        v_dev_view(n,i,j) = v_;

        mat_mul_dev_view(n,i,j) = 0.0;
        u_host_view(n,i,j)    = u_;
        v_host_view(n,i,j)    = v_;

        mat_mul_host_view(n,i,j) = 0.0;
        u_ptr_ok_host[IG(n,i,j)] = u_;
        v_ptr_ok_host[IG(n,i,j)] = v_;
    }

    for(std::size_t n=0u; n<N; ++n)
    for(std::size_t i=0u; i<K; ++i)
    for(std::size_t j=0u; j<K; ++j)
    {
        mat_mul_ptr_host[IC(n,i,j)] = 0.0;
        for(std::size_t k=0u; k < K; ++k)
            mat_mul_ptr_host[IC(n,i,j)] += u_ptr_host[IC(n,i,k)] * v_ptr_host[IC(n,k,j)];

        mat_mul_ptr_ok_host[IG(n,i,j)] = 0.0;
        for(std::size_t k=0u; k < K; ++k)
            mat_mul_ptr_ok_host[IG(n,i,j)] += u_ptr_ok_host[IG(n,i,k)] * v_ptr_ok_host[IG(n,k,j)];
    }

    __COMMON_PARTS_SAFE_CALL__( __COMMON_PARTS_DEVICE_MEMCPY__( (void*) u_ptr_func_dev,   (void*)u_ptr_ok_host,      sizeof(T)*total_size, __COMMON_PARTS_DEVICE_MEMCPY_HOST_TO_DEVICE__ ) );
    __COMMON_PARTS_SAFE_CALL__( __COMMON_PARTS_DEVICE_MEMCPY__( (void*) v_ptr_func_dev,   (void*)v_ptr_ok_host,      sizeof(T)*total_size, __COMMON_PARTS_DEVICE_MEMCPY_HOST_TO_DEVICE__ ) );
    __COMMON_PARTS_SAFE_CALL__( __COMMON_PARTS_DEVICE_MEMCPY__( (void*) u_ptr_dev,        (void*)u_ptr_host,         sizeof(T)*total_size, __COMMON_PARTS_DEVICE_MEMCPY_HOST_TO_DEVICE__ ) );
    __COMMON_PARTS_SAFE_CALL__( __COMMON_PARTS_DEVICE_MEMCPY__( (void*) v_ptr_dev,        (void*)v_ptr_host,         sizeof(T)*total_size, __COMMON_PARTS_DEVICE_MEMCPY_HOST_TO_DEVICE__ ) );
    __COMMON_PARTS_SAFE_CALL__( __COMMON_PARTS_DEVICE_MEMCPY__( (void*) u_ptr_ok_dev,     (void*)u_ptr_ok_host,      sizeof(T)*total_size, __COMMON_PARTS_DEVICE_MEMCPY_HOST_TO_DEVICE__ ) );
    __COMMON_PARTS_SAFE_CALL__( __COMMON_PARTS_DEVICE_MEMCPY__( (void*) v_ptr_ok_dev,     (void*)v_ptr_ok_host,      sizeof(T)*total_size, __COMMON_PARTS_DEVICE_MEMCPY_HOST_TO_DEVICE__ ) );

    u_dev_view.release(true);
    v_dev_view.release(true);
    mat_mul_dev_view.release(true);

    if((tests == 'd')||(tests == 'a'))
    {
	std::cout << "executing device tests ... " << std::endl;
        std::vector<double> gpu_tensor; gpu_tensor.reserve(number_of_iters);

        //WARM UP
        for(int it_ = 0; it_ < 5; it_++)
        {
            mat_mul_device_f<T, for_each_device_t, array_device_t>(N, u_dev, v_dev, mat_mul_dev);
        }
        device_e1.record();
        for(int it_ = 0; it_ < number_of_iters; it_++)
        {
            auto start = std::chrono::high_resolution_clock::now();
            mat_mul_device_f<T, for_each_device_t, array_device_t>(N, u_dev, v_dev, mat_mul_dev);
            __COMMON_PARTS_SAFE_CALL__( __COMMON_PARTS_DEVICE_SYNCRONIZE__() );
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> elapsed_seconds = end-start;
            gpu_tensor.push_back( elapsed_seconds.count() );
        }

        device_e2.record();
        std::cout << "device tensor time       = " <<  device_e2.elapsed_time(device_e1)/number_of_iters  << "ms." << std::endl;

        /***********************************************************************************************************/

        std::vector<double> gpu_ptr_func; gpu_ptr_func.reserve(number_of_iters);

        //WARM UP
        for(int it_ = 0; it_ < 5; it_++)
        {
            mat_mul_device_ptr<T, for_each_device_t>(N, u_ptr_func_dev, v_ptr_func_dev, mat_mul_ptr_func_dev);
        }
        device_e1.record();
        for(int it_ = 0; it_ < number_of_iters; it_++)
        {
            auto start = std::chrono::high_resolution_clock::now();
            mat_mul_device_ptr<T, for_each_device_t>(N, u_ptr_func_dev, v_ptr_func_dev, mat_mul_ptr_func_dev);
            __COMMON_PARTS_SAFE_CALL__( __COMMON_PARTS_DEVICE_SYNCRONIZE__() );
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> elapsed_seconds = end-start;
            gpu_ptr_func.push_back( elapsed_seconds.count() );
        }

        device_e2.record();
        std::cout << "device ptr_func time     = " <<  device_e2.elapsed_time(device_e1)/number_of_iters  << "ms." << std::endl;

        /**************************************************************************************************************/

        dim3 dimBlock(block_size,1);
        dim3 dimGrid( (N/block_size)+1,1);

        std::vector<double> gpu_ptr; gpu_ptr.reserve(number_of_iters);
        //WARM UP
        for(int it_ = 0; it_ < 5; it_++)
        {
             mat_mul_kern<T><<<dimGrid, dimBlock>>>(N, u_ptr_dev, v_ptr_dev, mat_mul_ptr_dev);
        }
        device_e1.record();
        for(int it_ = 0; it_ < number_of_iters; it_++)
        {
            auto start = std::chrono::high_resolution_clock::now();
            mat_mul_kern<T><<<dimGrid, dimBlock>>>(N, u_ptr_dev, v_ptr_dev, mat_mul_ptr_dev);
            __COMMON_PARTS_SAFE_CALL__( __COMMON_PARTS_DEVICE_SYNCRONIZE__() );
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> elapsed_seconds = end-start;
            gpu_ptr.push_back( elapsed_seconds.count() );
        }
        device_e2.record();
#ifdef USE_CONST
        std::cout << "device ptr_const time    = " <<  device_e2.elapsed_time(device_e1)/number_of_iters  << "ms." << std::endl;
#else
        std::cout << "device ptr time          = " <<  device_e2.elapsed_time(device_e1)/number_of_iters  << "ms." << std::endl;
#endif
        /***************************************************************************************************************/

        std::vector<double> gpu_ptr_ok; gpu_ptr_ok.reserve(number_of_iters);
        //WARM UP
        for(int it_ = 0; it_ < 5; it_++)
        {
            mat_mul_kern_ok<T><<<dimGrid, dimBlock>>>(N, u_ptr_ok_dev, v_ptr_ok_dev, mat_mul_ptr_ok_dev);
        }
        device_e1.record();
        for(int it_ = 0; it_ < number_of_iters; it_++)
        {
            auto start = std::chrono::high_resolution_clock::now();
            mat_mul_kern_ok<T><<<dimGrid, dimBlock>>>(N, u_ptr_ok_dev, v_ptr_ok_dev, mat_mul_ptr_ok_dev);
            __COMMON_PARTS_SAFE_CALL__( __COMMON_PARTS_DEVICE_SYNCRONIZE__() );
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> elapsed_seconds = end-start;
            gpu_ptr_ok.push_back( elapsed_seconds.count() );
        }

        device_e2.record();
#ifdef USE_CONST
        std::cout << "device ptr_ok_const time = " <<  device_e2.elapsed_time(device_e1)/number_of_iters  << "ms." << std::endl;
#else
        std::cout << "device ptr_ok time       = " <<  device_e2.elapsed_time(device_e1)/number_of_iters  << "ms." << std::endl;
#endif
        if(tests == 'a')
        {
        /***************************************************************************************************************/

            mat_mul_dev_view.init(mat_mul_dev, true);
            std::cout << "gpu tensor diff          = " << check_coincide_tensor(N, mat_mul_ptr_host, mat_mul_dev_view) << std::endl;
            mat_mul_dev_view.release(false);

            __COMMON_PARTS_SAFE_CALL__( __COMMON_PARTS_DEVICE_MEMCPY__( (void*) mat_mul_ptr_func_check, (void*)mat_mul_ptr_func_dev, sizeof(T)*total_size, __COMMON_PARTS_DEVICE_MEMCPY_DEVICE_TO_HOST__ ) );
            std::cout << "gpu ptr_func diff        = " << check_coincide_ptr(N, mat_mul_ptr_ok_host,  mat_mul_ptr_func_check) << std::endl;


            __COMMON_PARTS_SAFE_CALL__( __COMMON_PARTS_DEVICE_MEMCPY__( (void*) mat_mul_ptr_check, (void*)mat_mul_ptr_dev, sizeof(T)*total_size, __COMMON_PARTS_DEVICE_MEMCPY_DEVICE_TO_HOST__ ) );
#ifdef USE_CONST
            std::cout << "gpu ptr_const diff       = " << check_coincide_ptr(N, mat_mul_ptr_host, mat_mul_ptr_check) << std::endl;
#else
            std::cout << "gpu ptr    diff          = " << check_coincide_ptr(N, mat_mul_ptr_host, mat_mul_ptr_check) << std::endl;
#endif

            __COMMON_PARTS_SAFE_CALL__( __COMMON_PARTS_DEVICE_MEMCPY__( (void*) mat_mul_ptr_ok_check, (void*)mat_mul_ptr_ok_dev, sizeof(T)*total_size, __COMMON_PARTS_DEVICE_MEMCPY_DEVICE_TO_HOST__ ) );
#ifdef USE_CONST
            std::cout << "gpu ptr_ok_const diff    = " << check_coincide_ptr(N, mat_mul_ptr_ok_host, mat_mul_ptr_ok_check) << std::endl;
#else
            std::cout << "gpu ptr_ok diff          = " << check_coincide_ptr(N, mat_mul_ptr_ok_host, mat_mul_ptr_ok_check) << std::endl;
#endif
        /***************************************************************************************************************/
        }

        std::string filename;
        filename = "mat_mul_device_device_" + std::to_string(K) + "x" + std::to_string(K) + ".csv";
        std::fstream out_file{filename, out_file.out};
        if (!out_file.is_open())
            std::cout << "failed to open " << filename << '\n';
        else
        {
            out_file << "tensor,ptr_bad,ptr_ok" << std::endl;
            for(int j = 0; j<number_of_iters; j++)
            {
                out_file << gpu_tensor.at(j) << "," << gpu_ptr.at(j) << "," << gpu_ptr_ok.at(j) << std::endl;
            }
            out_file.close();
        }

    }
    if((tests == 'h')||(tests == 'a'))
    {


        std::cout << "executing host tests ... " << std::endl;
        std::vector<double> host_tensor; host_tensor.reserve(number_of_iters);
        std::vector<double> host_ptr; host_ptr.reserve(number_of_iters);
        std::vector<double> host_ptr_ok; host_ptr_ok.reserve(number_of_iters);

        host_e1.record();
        for(int it_ = 0; it_ < number_of_iters; it_++)
        {
            auto start = std::chrono::high_resolution_clock::now();
            mat_mul_host_f<T, for_each_omp_t, array_host_t>(N, u_host, v_host, mat_mul_host);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_seconds = end-start;
            host_tensor.push_back( elapsed_seconds.count() );

        }
        host_e2.record();
        std::cout << "host tensor time = " <<  host_e2.elapsed_time(host_e1)/number_of_iters  << "s." << std::endl;

        /**********************************************************************************************************/

        host_e1.record();
        for(int it_ = 0; it_ < number_of_iters; it_++)
        {
            auto start = std::chrono::high_resolution_clock::now();

            #pragma omp parallel for
            for(std::size_t n=0; n<N; ++n)
            {
                for(std::size_t i=0u; i<K; ++i)
                for(std::size_t j=0u; j<K; ++j)
                {
                    mat_mul_ptr_ok_host[IG(n,i,j)] = 0.f;
                    for(std::size_t k=0u; k<K; ++k)
                        mat_mul_ptr_ok_host[IG(n,i,j)] += u_ptr_ok_host[IG(n,i,k)] * v_ptr_ok_host[IG(n,k,j)];
                }
            }
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_seconds = end-start;
            host_ptr_ok.push_back( elapsed_seconds.count() );

        }
        host_e2.record();
        std::cout << "host ptr_ok time = " <<  host_e2.elapsed_time(host_e1)/number_of_iters  << "s." << std::endl;

        /*********************************************************************************************************/

        host_e1.record();
        for(int it_ = 0; it_ < number_of_iters; it_++)
        {
            auto start = std::chrono::high_resolution_clock::now();

            #pragma omp parallel for
            for(std::size_t n=0; n<N; ++n)
            {
                for(std::size_t i=0u; i<K; ++i)
                for(std::size_t j=0u; j<K; ++j)
                {
                    mat_mul_ptr_ok_host[IG(n,i,j)] = 0.f;
                    for(std::size_t k=0u; k<K; ++k)
                        mat_mul_ptr_host[IC(n,i,j)] += u_ptr_host[IC(n,i,k)] * v_ptr_host[IC(n,k,j)];
                }
            }

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_seconds = end-start;
            host_ptr.push_back( elapsed_seconds.count() );

        }
        host_e2.record();
        std::cout << "host ptr time    = " <<  host_e2.elapsed_time(host_e1)/number_of_iters  << "s." << std::endl;

        /*******************************************************************************************************/

        std::string filename;
        filename = "mat_mul_device_host_" + std::to_string(K) + "x" + std::to_string(K) + ".csv";
        std::fstream out_file_cpu{filename, out_file_cpu.out};
        if (!out_file_cpu.is_open())
            std::cout << "failed to open " << filename << '\n';
        else
        {
            out_file_cpu << "tensor,ptr_bad,ptr_ok" << std::endl;
            for(int j = 0; j<number_of_iters; j++)
            {
                out_file_cpu << host_tensor.at(j) << "," << host_ptr.at(j) << "," << host_ptr_ok.at(j) << std::endl;
            }
            out_file_cpu.close();
        }

    }

    std::free(u_ptr_ok_host);
    std::free(v_ptr_ok_host);

    std::free(mat_mul_ptr_host);
    std::free(mat_mul_ptr_ok_host);

    std::free(u_ptr_host);
    std::free(v_ptr_host);

    std::free(mat_mul_ptr_check);
    std::free(mat_mul_ptr_ok_check);
    std::free(mat_mul_ptr_func_check);

    __COMMON_PARTS_SAFE_CALL__( __COMMON_PARTS_DEVICE_FREE__(mat_mul_ptr_ok_dev) );
    __COMMON_PARTS_SAFE_CALL__( __COMMON_PARTS_DEVICE_FREE__(v_ptr_ok_dev) );
    __COMMON_PARTS_SAFE_CALL__( __COMMON_PARTS_DEVICE_FREE__(u_ptr_ok_dev) );

    __COMMON_PARTS_SAFE_CALL__( __COMMON_PARTS_DEVICE_FREE__(mat_mul_ptr_dev) );
    __COMMON_PARTS_SAFE_CALL__( __COMMON_PARTS_DEVICE_FREE__(v_ptr_dev) );
    __COMMON_PARTS_SAFE_CALL__( __COMMON_PARTS_DEVICE_FREE__(u_ptr_dev) );

    __COMMON_PARTS_SAFE_CALL__( __COMMON_PARTS_DEVICE_FREE__(mat_mul_ptr_func_dev) );
    __COMMON_PARTS_SAFE_CALL__( __COMMON_PARTS_DEVICE_FREE__(v_ptr_func_dev) );
    __COMMON_PARTS_SAFE_CALL__( __COMMON_PARTS_DEVICE_FREE__(u_ptr_func_dev) );


#endif    