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

#include <vector>
#include <algorithm>
#include <string>
#include <iostream>
#include <fstream>
#include <cstdio>
#include <cmath>
#include <limits>
#include <scfd/memory/host.h>
#include <scfd/static_vec/vec.h>
#include <scfd/static_vec/rect.h>
#include <scfd/communication/trivial_platform.h>
#include <scfd/communication/trivial_comm.h>
#include <scfd/communication/rect_partitioner.h>
#include <scfd/communication/trivial_binary_file.h>

const int Dim        = 3;
using mem_t           = scfd::memory::host;
using comm_t          = scfd::communication::trivial_platform<mem_t>;
using comm_info_type = scfd::communication::trivial_comm<mem_t>;

using ordinal     = std::int32_t;
using big_ordinal = std::ptrdiff_t;

using part_t    = scfd::communication::rect_partitioner<Dim, ordinal, big_ordinal, comm_info_type>;
using idx_t     = scfd::static_vec::vec<ordinal, Dim>;
using big_idx_t = scfd::static_vec::vec<big_ordinal, Dim>;

using big_ord_vec_t  = scfd::static_vec::vec<big_ordinal, Dim>;
using big_ord_rect_t = typename part_t::big_ord_rect_t;


struct shift //indexing temporal structure that returs indexes or seek position w.r.t. the first line.
{
    shift( const big_ord_vec_t &dom_size_p, const big_ord_rect_t &my_glob_rect_p )
        : dom_size_( dom_size_p ), my_own_glob_rect_( my_glob_rect_p )
    {
        Nx = dom_size_[0];
        Ny = dom_size_[1];
        Nz = dom_size_[2];
    }

    big_ordinal idx( const idx_t &i ) const
    {
        return (
            ( i( 2 ) + my_own_glob_rect_.i1( 2 ) ) * ( Ny ) * ( Nx ) + ( i( 1 ) + my_own_glob_rect_.i1( 1 ) ) * ( Nx ) +
            ( i( 0 ) + my_own_glob_rect_.i1( 0 ) )
        );
    }
    big_ordinal idx( const ordinal j, const ordinal k, const ordinal l ) const
    {
        return idx( { j, k, l } );
    }

    big_ord_vec_t idx3( const idx_t &i ) const
    {
        return {
            ( i( 0 ) + my_own_glob_rect_.i1( 0 ) ), ( i( 1 ) + my_own_glob_rect_.i1( 1 ) ),
            ( i( 2 ) + my_own_glob_rect_.i1( 2 ) )
        };
    }

    big_ordinal shft( const idx_t &i ) const
    {
        return (
            ( i( 2 ) + my_own_glob_rect_.i1( 2 ) ) * ( Ny ) * ( Nx ) + ( i( 1 ) + my_own_glob_rect_.i1( 1 ) ) * ( Nx ) +
            ( my_own_glob_rect_.i1( 0 ) )
        );
    }
    big_ordinal shft( const ordinal k, const ordinal l ) const
    {
        return shft( { 0, k, l } );
    }

    big_ord_rect_t my_own_glob_rect_;
    big_ord_vec_t  dom_size_;
    big_ordinal    Nx, Ny, Nz;
};


int main( int argc, char *argv[] )
{

    comm_t         comm( argc, argv );
    comm_info_type comm_info = comm.comm_world();
    auto           myproc    = comm_info.myid;

    int    Nx = 30, Ny = 47, Nz = 29;
    part_t part( comm_info, { Nx, Ny, Nz } );

    auto           domain         = part.get_dom_rect();
    auto           my_loc_rect    = part.get_own_loc_rect();
    auto           loc_dom_sz     = my_loc_rect.calc_size();
    big_ord_rect_t my_global_rect = part.get_own_rect();
    shift          sft( domain.calc_size(), my_global_rect );
    shift          sft_loc( loc_dom_sz, my_loc_rect );

    auto size_nd = my_global_rect.calc_size();
    auto n_loc   = my_global_rect.calc_area();


    std::vector<double> v( n_loc, myproc );
    for ( ordinal l = 0; l < loc_dom_sz( 2 ); l++ )
    {
        for ( ordinal k = 0; k < loc_dom_sz( 1 ); k++ )
        {
            for ( ordinal j = 0; j < loc_dom_sz( 0 ); j++ )
            {
                auto   big_idx = sft.idx3( { j, k, l } );
                double x       = ( static_cast<double>( big_idx( 0 ) ) - 0.5 * Nx ) / Nx;
                double y       = ( static_cast<double>( big_idx( 1 ) ) - 0.5 * Ny ) / Ny;
                double z       = ( static_cast<double>( big_idx( 2 ) ) - 0.5 * Nz ) / Nz;

                auto loc_idx = sft_loc.idx( { j, k, l } );
                v[loc_idx]   = exp( 10.0 * ( -x * x - y * y - z * z ) ); //filler function
            }
        }
    }
    std::vector<double> v_ref;
    std::copy( v.begin(), v.end(), std::back_inserter( v_ref ) );

    std::string file_name{ "test_trivial_file_write.txt" };
    { //writing file
        scfd::communication::trivial_binary_file<double> file( comm_info, file_name );
        file.size( domain.calc_area() ); //instead of preallocate

        std::vector<double> buf( loc_dom_sz( 0 ), 0 );
        for ( ordinal l = 0; l < loc_dom_sz( 2 ); l++ )
        {
            for ( ordinal k = 0; k < loc_dom_sz( 1 ); k++ )
            {
                for ( ordinal j = 0; j < loc_dom_sz( 0 ); j++ )
                {
                    auto loc_idx = sft_loc.idx( { j, k, l } );
                    buf[j]       = v[loc_idx];
                }
                auto shft = sft.shft( k, l );
                file.write_at( shft, buf.data(), loc_dom_sz( 0 ) );
            }
        }
    }
    { //reading file
        scfd::communication::trivial_binary_file<double> file(
            comm_info, file_name, scfd::communication::binary_file_mode::rdonly
        );
        std::vector<double> buf( loc_dom_sz( 0 ), 0 );

        for ( ordinal l = 0; l < loc_dom_sz( 2 ); l++ )
        {
            for ( ordinal k = 0; k < loc_dom_sz( 1 ); k++ )
            {
                auto shft = sft.shft( k, l );
                file.seek( shft );
                file.read( loc_dom_sz( 0 ), buf.data() );
                for ( ordinal j = 0; j < loc_dom_sz( 0 ); j++ )
                {
                    auto loc_idx = sft_loc.idx( { j, k, l } );
                    v[loc_idx]   = buf[j];
                }
            }
        }
    }

    bool is_failed = false;
    for ( ordinal j = 0; j < n_loc; j++ )
    {
        if ( std::abs( v_ref[j] - v[j] ) > std::numeric_limits<double>::epsilon() )
        {
            std::cerr << "error at: " << j << " with ref = " << v_ref[j] << " and file read = " << v[j] << std::endl;
            is_failed = true;
        }
    }
    std::remove( file_name.c_str() );

    std::string size_file_name{ "test_trivial_file_size.txt" };
    {
        scfd::communication::trivial_binary_file<double> file( comm_info, size_file_name );
        file.size( 100 );
        std::ifstream check( size_file_name, std::ios::binary | std::ios::ate );
        auto           grown_sz = static_cast<std::streamoff>( check.tellg() );
        check.close();
        if ( grown_sz != static_cast<std::streamoff>( 100 * sizeof( double ) ) )
        {
            std::cerr << "error: size() did not grow file, got " << grown_sz << std::endl;
            is_failed = true;
        }
    }
    {
        scfd::communication::trivial_binary_file<double> file( comm_info, size_file_name );
        file.size( 10 );
        std::ifstream check( size_file_name, std::ios::binary | std::ios::ate );
        auto           shrunk_sz = static_cast<std::streamoff>( check.tellg() );
        check.close();
        if ( shrunk_sz != static_cast<std::streamoff>( 10 * sizeof( double ) ) )
        {
            std::cerr << "error: size() did not shrink file, got " << shrunk_sz << std::endl;
            is_failed = true;
        }
    }
    std::remove( size_file_name.c_str() );

    std::string wat_file_name{ "test_trivial_file_write_at.txt" };
    {
        scfd::communication::trivial_binary_file<double> file( comm_info, wat_file_name );
        file.size( 20 );

        std::vector<double> wbuf( 5 );
        for ( int i = 0; i < 5; i++ )
        {
            wbuf[i] = 3.14159 * i + 1.0;
        }
        file.write_at( 7, wbuf.data(), 5 );

        std::vector<double> rbuf( 5, 0.0 );
        file.read_at( 7, 5, rbuf.data() );

        for ( int i = 0; i < 5; i++ )
        {
            if ( std::abs( wbuf[i] - rbuf[i] ) > std::numeric_limits<double>::epsilon() )
            {
                std::cerr << "error at write_at/read_at: " << i << " with ref = " << wbuf[i]
                           << " and file read = " << rbuf[i] << std::endl;
                is_failed = true;
            }
        }
    }
    std::remove( wat_file_name.c_str() );

    if ( is_failed )
    {
        std::cout << "FAILED" << std::endl;
    }
    else
    {
        std::cout << "PASSED" << std::endl;
    }

    return is_failed;
}
