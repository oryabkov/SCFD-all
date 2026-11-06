// Copyright © 2023-2025 Ryabkov Oleg Igorevich, Evstigneev Nikolay Mikhaylovitch

// This file is part of SCFD.

// SCFD is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 2 only of the License.

// SCFD is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with SCFD.  If not, see <http://www.gnu.org/licenses/>.

#ifndef __SCFD_COMM_TRIVIAL_BINARY_FILE_H__
#define __SCFD_COMM_TRIVIAL_BINARY_FILE_H__

#include <string>
#include <fstream>
#include <stdexcept>
#include <cstdint>
#include <cstddef>
#include <sys/types.h>
#include <unistd.h>


namespace scfd
{
namespace communication
{

struct binary_file_mode
{
    static const int rdonly = 1 << 0;
    static const int wronly = 1 << 1;
    static const int rdwr   = 1 << 2;
    static const int create = 1 << 3;
    static const int excl   = 1 << 4;
    static const int append = 1 << 5;
};

struct binary_file_seek
{
    static const int set = 0;
    static const int cur = 1;
    static const int end = 2;
};

template <class T>
struct trivial_binary_file
{
    using offset_type = long long;

    template <class Comm>
    trivial_binary_file(
        const Comm &comm_info, const std::string &file_name, int amode = binary_file_mode::rdwr | binary_file_mode::create
    ) : file_name_( file_name )
    {
        (void)comm_info;

        bool want_create = ( amode & binary_file_mode::create ) != 0;
        bool want_excl   = ( amode & binary_file_mode::excl ) != 0;
        if ( want_create && want_excl )
        {
            std::ifstream probe( file_name_, std::ios_base::binary );
            if ( probe.good() )
            {
                throw std::runtime_error( "trivial_binary_file: file already exists: " + file_name_ );
            }
        }

        bool want_read  = ( amode & binary_file_mode::rdonly ) != 0 || ( amode & binary_file_mode::rdwr ) != 0;
        bool want_write =
            ( amode & binary_file_mode::wronly ) != 0 || ( amode & binary_file_mode::rdwr ) != 0 || want_create;

        std::ios_base::openmode mode = std::ios_base::binary;
        if ( want_write )
        {
            mode |= std::ios_base::in | std::ios_base::out;
        }
        else if ( want_read )
        {
            mode |= std::ios_base::in;
        }
        if ( ( amode & binary_file_mode::append ) != 0 )
        {
            mode |= std::ios_base::app;
        }

        fs_.open( file_name_, mode );
        if ( !fs_.is_open() && want_create )
        {
            std::fstream creator( file_name_, std::ios_base::out | std::ios_base::binary );
            creator.close();
            fs_.clear();
            fs_.open( file_name_, mode );
        }
        if ( !fs_.is_open() )
        {
            throw std::runtime_error( "trivial_binary_file: failed to open file: " + file_name_ );
        }
    }
    ~trivial_binary_file()
    {
        fs_.close();
    }
    void seek( offset_type offset, int whence = binary_file_seek::set ) const
    {
        std::ios_base::seekdir dir = std::ios_base::beg;
        if ( whence == binary_file_seek::cur )
        {
            dir = std::ios_base::cur;
        }
        else if ( whence == binary_file_seek::end )
        {
            dir = std::ios_base::end;
        }
        fs_.seekg( offset * sizeof( T ), dir );
        fs_.seekp( offset * sizeof( T ), dir );
    }
    void size( offset_type size ) const
    {
        fs_.flush();
        ::truncate( file_name_.c_str(), static_cast<off_t>( size * sizeof( T ) ) );
        fs_.clear();
    }
    void preallocate( offset_type size ) const
    {
        fs_.flush();
        std::ifstream probe( file_name_, std::ios_base::binary | std::ios_base::ate );
        offset_type   target = size * sizeof( T );
        if ( !probe.good() || probe.tellg() < static_cast<std::streamoff>( target ) )
        {
            ::truncate( file_name_.c_str(), static_cast<off_t>( target ) );
        }
        fs_.clear();
    }
    void write( const T *data, int count ) const
    {
        fs_.write( reinterpret_cast<const char *>( data ), static_cast<std::streamsize>( count ) * sizeof( T ) );
    }
    void write_at( offset_type offset, const T *data, int count ) const
    {
        fs_.seekp( offset * sizeof( T ), std::ios_base::beg );
        write( data, count );
    }
    void read( int count, T *data ) const
    {
        fs_.read( reinterpret_cast<char *>( data ), static_cast<std::streamsize>( count ) * sizeof( T ) );
    }
    void read_at( offset_type offset, int count, T *data ) const
    {
        fs_.seekg( offset * sizeof( T ), std::ios_base::beg );
        read( count, data );
    }

    std::fstream &handle()
    {
        return fs_;
    }

private:
    std::string           file_name_;
    mutable std::fstream  fs_;
};


} // namespace communication
} // namespace scfd

#endif
