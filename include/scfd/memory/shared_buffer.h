#include <stdexcept>

namespace scfd
{
namespace memory
{

template <class Memory>
class shared_buffer
{
public:
    using pointer_t = typename Memory::pointer_type;


    explicit shared_buffer()
        : work_area_size_( 0 ), work_area_( nullptr ), taken_( false ), work_area_size_previous_( 0 )
    {
    }

    ~shared_buffer()
    {
        try
        {
            Memory::free( work_area_ );
            work_area_size_previous_ = 0;
            work_area_size_          = 0;
        }
        catch ( ... )
        {
        }
    }


    void require_size_bytes( const std::size_t reqired_bytes )
    {
        work_area_size_ = ( work_area_size_ < reqired_bytes ) ? reqired_bytes : work_area_size_;
    }

    std::size_t get_work_size() const
    {
        return work_area_size_;
    }

    void activate()
    {
        if ( work_area_size_previous_ == 0 )
        {
            Memory::malloc( &work_area_, work_area_size_ );
        }
        else if ( work_area_size_ > work_area_size_previous_ )
        {
            Memory::free( work_area_ );
            Memory::malloc( &work_area_, work_area_size_ );
        }
        work_area_size_previous_ = work_area_size_;
    }

    pointer_t naive_ptr() const
    {
        return work_area_;
    }

    void take() const
    {
        if ( ( taken_ ) || ( work_area_size_previous_ == 0 ) )
        {
            throw std::logic_error( "shared_buffer::take: the buffer ref is already taken or not activated" );
        }
        taken_ = true;
    }

    void release() const
    {
        if ( ( !taken_ ) || ( work_area_size_previous_ == 0 ) )
        {
            throw std::logic_error( "shared_buffer::release: the buffer ref is not taken or not activated." );
        }
        taken_ = false;
    }
    bool is_taken() const
    {
        return taken_;
    }

private:
    std::size_t  work_area_size_, work_area_size_previous_;
    pointer_t    work_area_;
    mutable bool taken_;
};


}
}