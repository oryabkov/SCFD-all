#ifndef __ALL_KERNELS_H__
#define __ALL_KERNELS_H__

/**
 * 
 * This header file lists all common kernels that are independent on HIP or CUDA reserved words
 */

/******************************* BEGIN DEVICE KERNELS ***************************************/
template<class T>
#ifdef USE_CONST
__global__ void mat_mul_kern(std::size_t N, const T* f1_, const T* f2_, T* f_out_)
#else
__global__ void mat_mul_kern(std::size_t N, T* f1_, T* f2_, T* f_out_)
#endif
{
    T buf[K];
    T val_l;
    SCFD_ARRAYS_ORDINAL_TYPE idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx>=N) return;
    #pragma unroll
    for(int i = 0; i < K; ++i)
    {
        #pragma unroll
        for(int j = 0; j < K; ++j)
        {
            val_l = static_cast<T>(0.0);
            #pragma unroll
            for(int k = 0; k < K; ++k)
            {
                val_l = fma(f1_[IC(idx, i, k)], f2_[IC(idx, k, j)], val_l);
            }
            buf[j] = val_l;
        }
        #pragma unroll
        for(int k = 0; k < K; ++k)
        {
            f_out_[IC(idx, i, k)] = buf[k];
        }
    }    

}

template<class T>
#ifdef USE_CONST
__global__ void mat_mul_kern_ok(std::size_t N, const T* f1_, const T* f2_, T* f_out_)
#else
__global__ void mat_mul_kern_ok(std::size_t N, T* f1_, T* f2_, T* f_out_)
#endif
{
    T buf[K];
    T val_l;
    SCFD_ARRAYS_ORDINAL_TYPE idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx>=N) return;
    #pragma unroll
    for(int i = 0; i < K; ++i)
    {
        #pragma unroll
        for(int j = 0; j < K; ++j)
        {
            val_l = static_cast<T>(0.0);
            #pragma unroll
            for(int k = 0; k < K; ++k)
            {
                val_l = fma(f1_[IG(idx, i, k)], f2_[IG(idx, k, j)], val_l);
            }
            buf[j] = val_l;
        }
        #pragma unroll
        for(int k = 0; k < K; ++k)
        {
            f_out_[IG(idx, i, k)] = buf[k];
        }
    }
}

template<class T>
#ifdef USE_CONST
__global__ void mat_mul_kern_ok_mm(std::size_t N, const T* f1_, const T* f2_, T* f_out_)
#else
__global__ void mat_mul_kern_ok_mm(std::size_t N, T* f1_, T* f2_, T* f_out_)
#endif
{
    T mat1[K][K];
    T mat2[K][K];
    T mat3[K][K];
    T val_l;
    SCFD_ARRAYS_ORDINAL_TYPE idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx>=N) return;
    
    #pragma unroll
    for(int i = 0; i < K; ++i)
    {
        #pragma unroll
        for(int j = 0; j < K; ++j)
        {
            mat1[i][j] = f1_[IG(idx, i, j)];
            mat2[i][j] = f2_[IG(idx, i, j)];
        }
    }

    #pragma unroll
    for(int i = 0; i < K; ++i)
    {
        #pragma unroll
        for(int j = 0; j < K; ++j)
        {
            val_l = static_cast<T>(0.0);
            #pragma unroll
            for(int k = 0; k < K; ++k)
            {
                val_l = fma(mat1[i][k], mat2[k][j], val_l);
            }
            mat3[i][j] = val_l;
        }
    }
    #pragma unroll
    for(int i = 0; i < K; ++i)
    {
        #pragma unroll
        for(int j = 0; j < K; ++j)
        {
            f_out_[IG(idx, i, j)] = mat3[i][j];
        }
    }

}


/******************************* END DEVICE KERNELS ***************************************/



/****************************** BEGIN TENSOR FUNCTOR **************************************/
template<class T, class MatrixND>
struct func_mat_mul
{

    MatrixND f1_;
    MatrixND f2_;
    MatrixND f_out_;

    func_mat_mul(const MatrixND &f1, const MatrixND &f2, MatrixND &f_out):
    f1_(f1),
    f2_(f2),
    f_out_(f_out)
    {}
    __DEVICE_TAG__ void operator()(const int &idx)
    {
        T buf[K];
        T val_l;
        #pragma unroll
        for(int i = 0; i < K; ++i)
        {
            #pragma unroll
            for(int j = 0; j < K; ++j)
            {
                val_l = static_cast<T>(0.0);
                #pragma unroll
                for(int k = 0; k < K; ++k)
                {
                    val_l = fma(f1_(idx, i, k), f2_(idx, k, j), val_l);
                }
                buf[j] = val_l;
            }
            #pragma unroll
            for(int k = 0; k < K; ++k)
            {
                f_out_(idx, i, k) = buf[k];
            }
        }        
    }
};


template<class T, class MatrixND>
struct func_mat_mul_mm
{

    MatrixND f1_;
    MatrixND f2_;
    MatrixND f_out_;

    func_mat_mul_mm(const MatrixND &f1, const MatrixND &f2, MatrixND &f_out):
    f1_(f1),
    f2_(f2),
    f_out_(f_out)
    {}
    __DEVICE_TAG__ void operator()(const int &idx)
    {
        T mat1[K][K];
        T mat2[K][K];
        T mat3[K][K];
        T val_l;
        
        #pragma unroll
        for(int i = 0; i < K; ++i)
        {
            #pragma unroll
            for(int j = 0; j < K; ++j)
            {
                mat1[i][j] = f1_(idx, i, j);
                mat2[i][j] = f2_(idx, i, j);
            }
        }
        #pragma unroll
        for(int i = 0; i < K; ++i)
        {
            #pragma unroll
            for(int j = 0; j < K; ++j)
            {
                val_l = static_cast<T>(0.0);
                #pragma unroll
                for(int k = 0; k < K; ++k)
                {
                    val_l = fma(mat1[i][k], mat2[k][j], val_l);
                }
                mat3[i][j] = val_l;
            }
        }
        #pragma unroll
        for(int i = 0; i < K; ++i)
        {
            #pragma unroll
            for(int j = 0; j < K; ++j)
            {
                f_out_(idx, i, j) = mat3[i][j];
            }
        }        
    }
};


template<class T, class ForEach, class MatrixND>
void mat_mul_device_f(const std::size_t N, const MatrixND& u, const MatrixND& v, MatrixND& w)
{

    ForEach for_each;
    for_each.block_size = block_size;
    for_each(func_mat_mul<T, MatrixND>(u, v, w), 0, N);
    for_each.wait();

}
template<class T, class ForEach, class MatrixND>
void mat_mul_device_mm_f(const std::size_t N, const MatrixND& u, const MatrixND& v, MatrixND& w)
{

    ForEach for_each;
    for_each.block_size = block_size;
    for_each(func_mat_mul_mm<T, MatrixND>(u, v, w), 0, N);
    for_each.wait();

}
template<class T, class ForEach, class MatrixND>
void mat_mul_host_f(const std::size_t N, const MatrixND& u, const MatrixND& v, MatrixND& w)
{

    ForEach for_each;
    for_each(func_mat_mul<T, MatrixND>(u, v, w), 0, N);
    for_each.wait();

}
/******************************* END TENSOR FUNCTOR ***************************************/



/******************************** BEGIN PTR FUNCTOR ***************************************/
template<class T>
struct func_mat_mul_ptr
{

    const T* f1_;
    const T* f2_;
    T*    f_out_;
    std::size_t N;

    func_mat_mul_ptr(std::size_t const sz, const T* f1, const T* f2, T* f_out):
    N(sz),
    f1_(f1),
    f2_(f2),
    f_out_(f_out)
    {}


    __DEVICE_TAG__ void operator()(const int &idx)
    {
        T buf1[K];
        T buf2[K];
        T buf3[K];   
        T val_l;     
        for(std::size_t i = 0u; i < K; ++i)
        {
            for(std::size_t j = 0u; j < K; ++j)
            {
                #pragma unroll
                for(std::size_t k = 0u; k < K; ++k)
                {
                    buf1[k] = f1_[IG(idx, i, k)];
                    buf2[k] = f2_[IG(idx, k, j)];
                }
                val_l = static_cast<T>(0.0);
                #pragma unroll
                for(std::size_t k = 0u; k < K; ++k)
                {
                    val_l = fma(buf1[k], buf2[k], val_l);
                }
                buf3[j] = val_l;
            }
            #pragma unroll
            for(std::size_t k = 0u; k < K; ++k)
            {
                f_out_[IG(idx, i, k)] = buf3[k];
            }
        }
    }
};

template<class T, class ForEach>
void mat_mul_device_ptr(const std::size_t N, const T* u, const T* v, T* w)
{

    ForEach for_each;
    for_each.block_size = block_size;
    for_each(func_mat_mul_ptr<T>(N, u, v, w), 0, N);
    for_each.wait();

}
template<class T, class ForEach>
void mat_mul_host_ptr(const std::size_t N, const T* u, const T* v, T* w)
{

    ForEach for_each;
    for_each(func_mat_mul_ptr<T>(N, u, v, w), 0, N);
    for_each.wait();

}
/******************************** END PTR FUNCTOR *****************************************/



template<class T, class MatrixND>
T check_coincide_tensor(const std::size_t N, const T* ptr_, const MatrixND& ta_)
{
    T diff = 0.0;

    for(std::size_t n=0; n<N; ++n)
    for(std::size_t i=0; i<K; ++i)
    for(std::size_t j=0; j<K; ++j)
        diff += std::abs(ptr_[IC(n,i,j)] - ta_(n,i,j) );

    return diff;
}

template<class T>
T check_coincide_ptr(std::size_t N, const T* ptr_, const T* ta_)
{
    T diff = 0.0;

    for(std::size_t n=0; n<N; ++n)
    for(std::size_t i=0; i<K; ++i)
    for(std::size_t j=0; j<K; ++j)
        diff += std::abs(ptr_[IC(n,i,j)] - ta_[IC(n,i,j)] );

    return diff;
}

#endif // ALL_KERNELS_H