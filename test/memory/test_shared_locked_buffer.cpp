#include <iostream>
#include <stdexcept>
#include <iomanip>
#include <scfd/memory/host.h>
#include <scfd/memory/shared_buffer.h>

template <class Buf>
void check_ptr_bytes( const Buf &buf )
{
    {
        buf.take();
        int *ptr = static_cast<int *>( buf.naive_ptr() );
        for ( int i = 0; i < buf.get_work_size() / sizeof( int ); i++ )
        {
            ptr[i] = static_cast<int>( i );
        }
        buf.release();
    }
    {
        buf.take();
        auto ptr  = static_cast<int *>( buf.naive_ptr() );
        int  diff = 0;
        for ( int i = 0; i < buf.get_work_size() / sizeof( int ); i++ )
        {
            diff += static_cast<int>( ptr[i] ) - static_cast<int>( i );
        }
        buf.release();
        if ( diff > 0 )
        {
            throw std::runtime_error( "incorrect difference in array check" );
        }
    }
}


int main( int argc, char const *argv[] )
{
    using T     = double;
    using mem_t = scfd::memory::host;
    using buf_t = scfd::memory::shared_buffer<mem_t>;

    int   failed = 0;
    buf_t buf;
    try
    {
        buf.require_size_bytes( 140 * sizeof( T ) );
        buf.require_size_bytes( 250 * sizeof( std::uint8_t ) );
        buf.require_size_bytes( 1100 );
        buf.activate();
        auto ptr_address = buf.naive_ptr();
        std::cout << "work size: " << buf.get_work_size() << "B, ptr adress: " << std::hex << "0x"
                  << reinterpret_cast<std::ptrdiff_t>( ptr_address ) << std::dec << std::endl;

        check_ptr_bytes( buf );
        buf.require_size_bytes( 1100 * sizeof( T ) );
        buf.require_size_bytes( 3000 );
        buf.activate();
        std::cout << "reactivated work size: " << buf.get_work_size() << "B, new ptr adress: " << std::hex << "0x"
                  << reinterpret_cast<std::ptrdiff_t>( buf.naive_ptr() ) << std::dec << std::endl;
        check_ptr_bytes( buf );
    }
    catch ( const std::exception &e )
    {
        std::cerr << "cought exception: " << e.what() << std::endl;
        failed = 1;
    }

    std::cout << "checking exceptions: " << std::endl;
    try
    {
        buf.take();
        buf.take();
        failed == 1;
    }
    catch ( const std::exception &e )
    {
        std::cerr << "    - cought correct exception: " << e.what() << std::endl;
    }
    try
    {
        buf.release();
        buf.release();
        failed == 1;
    }
    catch ( const std::exception &e )
    {
        std::cerr << "    - cought correct exception: " << e.what() << std::endl;
    }
    if ( buf.is_taken() )
    {
        failed = 1;
        std::cerr << "buffer is taken, while it should not!" << std::endl;
    }
    if ( failed == 1 )
    {
        std::cerr << "FAILED" << std::endl;
    }
    else
    {
        std::cout << "PASS" << std::endl;
    }
    return failed;
}