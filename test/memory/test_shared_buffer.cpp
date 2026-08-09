#include <iostream>
#include <stdexcept>
#include <iomanip>
#include <cstdint>
#include <cstdlib>
#include <scfd/memory/host.h>
#include <scfd/memory/shared_buffer.h>

struct tracked_host_memory
{
    typedef tracked_host_memory host_memory_type;
    typedef void               *pointer_type;
    typedef const void         *const_pointer_type;

    static const bool is_host_visible         = true;
    static const bool prefer_array_of_structs = true;

    static int         malloc_calls;
    static int         free_calls;
    static std::size_t last_malloc_size;

    static void reset()
    {
        malloc_calls     = 0;
        free_calls       = 0;
        last_malloc_size = 0;
    }

    static void malloc( pointer_type *p, std::size_t size )
    {
        ++malloc_calls;
        last_malloc_size = size;
        scfd::memory::host::malloc( p, size );
    }

    static void free( pointer_type p )
    {
        if ( p != nullptr )
        {
            ++free_calls;
        }
        scfd::memory::host::free( p );
    }
};

int         tracked_host_memory::malloc_calls     = 0;
int         tracked_host_memory::free_calls       = 0;
std::size_t tracked_host_memory::last_malloc_size = 0;

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

void check_take_true_reallocates_after_size_increase()
{
    typedef scfd::memory::shared_buffer<tracked_host_memory> tracked_buf_t;

    const std::size_t small_size = 128;
    const std::size_t large_size = 4096;

    tracked_host_memory::reset();

    {
        tracked_buf_t buf;
        buf.require_size_bytes( small_size );
        buf.activate();

        if ( tracked_host_memory::malloc_calls != 1 )
        {
            throw std::runtime_error( "initial activate did not allocate exactly once" );
        }
        if ( tracked_host_memory::last_malloc_size != small_size )
        {
            throw std::runtime_error( "initial activate allocated incorrect size" );
        }

        buf.require_size_bytes( large_size );
        buf.take( true );

        if ( !buf.is_taken() )
        {
            throw std::runtime_error( "take(true) did not mark buffer as taken after reallocation" );
        }
        if ( tracked_host_memory::free_calls != 1 )
        {
            throw std::runtime_error( "take(true) did not free previous allocation after size increase" );
        }
        if ( tracked_host_memory::malloc_calls != 2 )
        {
            throw std::runtime_error( "take(true) did not allocate updated buffer after size increase" );
        }
        if ( tracked_host_memory::last_malloc_size != large_size )
        {
            throw std::runtime_error( "take(true) allocated incorrect updated size" );
        }

        char *ptr           = static_cast<char *>( buf.naive_ptr() );
        ptr[0]              = 1;
        ptr[large_size - 1] = 2;

        buf.release();
    }

    if ( tracked_host_memory::free_calls != 2 )
    {
        throw std::runtime_error( "shared_buffer destructor did not release updated allocation" );
    }
}

template <class Buf>
void check_const_take_on_activated_buffer( const Buf &buf )
{
    buf.take();
    if ( !buf.is_taken() )
    {
        throw std::runtime_error( "const take did not mark activated buffer as taken" );
    }

    buf.release();
    if ( buf.is_taken() )
    {
        throw std::runtime_error( "release did not reset buffer taken state after const take" );
    }
}

template <class Buf>
void check_const_take_requires_activation()
{
    Buf        buf;
    const Buf &const_buf          = buf;
    bool       failed_as_expected = false;

    try
    {
        const_buf.take();
    }
    catch ( const std::exception & )
    {
        failed_as_expected = true;
    }

    if ( !failed_as_expected )
    {
        throw std::runtime_error( "const take succeeded on non-activated buffer" );
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
        check_const_take_on_activated_buffer( buf );
        buf.require_size_bytes( 1100 * sizeof( T ) );
        buf.require_size_bytes( 3000 );
        buf.activate();
        std::cout << "reactivated work size: " << buf.get_work_size() << "B, new ptr adress: " << std::hex << "0x"
                  << reinterpret_cast<std::ptrdiff_t>( buf.naive_ptr() ) << std::dec << std::endl;
        check_ptr_bytes( buf );
        check_const_take_on_activated_buffer( buf );
        check_const_take_requires_activation<buf_t>();
        check_take_true_reallocates_after_size_increase();
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
        failed = 1;
    }
    catch ( const std::exception &e )
    {
        std::cerr << "    - cought correct exception: " << e.what() << std::endl;
    }
    try
    {
        buf.release();
        buf.release();
        failed = 1;
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
