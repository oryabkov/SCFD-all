#ifndef __SCFD_LAPACK_WRAP_DEVICE_H__
#define __SCFD_LAPACK_WRAP_DEVICE_H__

#include <cstddef>
#include <string>
#include <vector>

#include <scfd/external_libraries/lapack_wrap.h>

namespace scfd
{

template <class Backend, class T>
class lapack_wrap_device : public lapack_wrap<T>
{
public:
    using backend_type = Backend;
    using memory_type = typename backend_type::memory_type;
    using host_lapack_type = lapack_wrap<T>;

    explicit lapack_wrap_device( size_t expected_size ) : host_lapack_type( expected_size )
    {
    }

    void hessinberg_eigs_from_device( const T *H_device, size_t Nl, T *eig_real, T *eig_imag )
    {
        host_lapack_type::hessinberg_eigs( host_matrix( H_device, Nl * Nl ), Nl, eig_real, eig_imag );
    }

    template <class C_t>
    void hessinberg_eigs_from_device( const T *H_device, size_t Nl, C_t *eig )
    {
        host_lapack_type::hessinberg_eigs( host_matrix( H_device, Nl * Nl ), Nl, eig );
    }

    void hessinberg_schur_from_device(
        const T *H_device, size_t Nl, T *Q, T *R, T *eig_real = nullptr, T *eig_imag = nullptr
    )
    {
        host_lapack_type::hessinberg_schur( host_matrix( H_device, Nl * Nl ), Nl, Q, R, eig_real, eig_imag );
    }

    template <class C_t>
    void hessinberg_schur_from_device( const T *H_device, size_t Nl, T *Q, T *R, C_t *eig )
    {
        host_lapack_type::hessinberg_schur( host_matrix( H_device, Nl * Nl ), Nl, Q, R, eig );
    }

    void eigs_schur_from_device(
        const T *A_device, size_t Nl, T *Q, T *R, T *eig_real = nullptr, T *eig_imag = nullptr
    )
    {
        host_lapack_type::eigs_schur( host_matrix( A_device, Nl * Nl ), Nl, eig_real, eig_imag, Q, R );
    }

    template <class C_t>
    void eigs_schur_from_device( const T *A_device, size_t Nl, T *Q, T *R, C_t *eig )
    {
        host_lapack_type::eigs_schur( host_matrix( A_device, Nl * Nl ), Nl, eig, Q, R );
    }

    template <class T_l>
    void write_matrix_from_device( const std::string &f_name, size_t Row, size_t Col, T_l *matrix, unsigned int prec = 17 )
    {
        if constexpr ( memory_type::is_host_visible )
        {
            host_lapack_type::write_matrix( f_name, Row, Col, matrix, prec );
        }
        else
        {
            std::vector<T_l> host_matrix_l( Row * Col );
            copy_to_host( matrix, host_matrix_l.data(), Row * Col );
            host_lapack_type::write_matrix( f_name, Row, Col, host_matrix_l.data(), prec );
        }
    }

    template <class T_l>
    void write_vector_from_device( const std::string &f_name, size_t N, T_l *vec, unsigned int prec = 17 )
    {
        if constexpr ( memory_type::is_host_visible )
        {
            host_lapack_type::write_vector( f_name, N, vec, prec );
        }
        else
        {
            std::vector<T_l> host_vec_l( N );
            copy_to_host( vec, host_vec_l.data(), N );
            host_lapack_type::write_vector( f_name, N, host_vec_l.data(), prec );
        }
    }

    void hessinberg_eigs_from_gpu( const T *H_device, size_t Nl, T *eig_real, T *eig_imag )
    {
        hessinberg_eigs_from_device( H_device, Nl, eig_real, eig_imag );
    }

    template <class C_t>
    void hessinberg_eigs_from_gpu( const T *H_device, size_t Nl, C_t *eig )
    {
        hessinberg_eigs_from_device( H_device, Nl, eig );
    }

    void hessinberg_schur_from_gpu(
        const T *H_device, size_t Nl, T *Q, T *R, T *eig_real = nullptr, T *eig_imag = nullptr
    )
    {
        hessinberg_schur_from_device( H_device, Nl, Q, R, eig_real, eig_imag );
    }

    template <class C_t>
    void hessinberg_schur_from_gpu( const T *H_device, size_t Nl, T *Q, T *R, C_t *eig )
    {
        hessinberg_schur_from_device( H_device, Nl, Q, R, eig );
    }

    void eigs_schur_from_gpu( const T *A_device, size_t Nl, T *Q, T *R, T *eig_real = nullptr, T *eig_imag = nullptr )
    {
        eigs_schur_from_device( A_device, Nl, Q, R, eig_real, eig_imag );
    }

    template <class C_t>
    void eigs_schur_from_gpu( const T *A_device, size_t Nl, T *Q, T *R, C_t *eig )
    {
        eigs_schur_from_device( A_device, Nl, Q, R, eig );
    }

private:
    const T *host_matrix( const T *device_ptr, size_t count )
    {
        if constexpr ( memory_type::is_host_visible )
        {
            return device_ptr;
        }
        else
        {
            host_matrix_buffer_.resize( count );
            copy_to_host( device_ptr, host_matrix_buffer_.data(), count );
            return host_matrix_buffer_.data();
        }
    }

    template <class T_l>
    static void copy_to_host( const T_l *device_ptr, T_l *host_ptr, size_t count )
    {
        memory_type::copy_to_host(
            count * sizeof( T_l ),
            static_cast<typename memory_type::const_pointer_type>( device_ptr ),
            static_cast<typename memory_type::pointer_type>( host_ptr )
        );
    }

    std::vector<T> host_matrix_buffer_;
};

} // namespace scfd

#endif
