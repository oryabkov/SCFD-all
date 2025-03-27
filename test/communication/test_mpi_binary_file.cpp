#include <vector>
#include <algorithm>
#include <string>
#include <iostream>
#include <cmath>
#include <scfd/static_vec/vec.h>
#include <scfd/static_vec/rect.h>
#include <scfd/communication/mpi_wrap.h>
#include <scfd/communication/mpi_comm_info.h>
#include <scfd/communication/rect_partitioner.h>
#include <scfd/communication/mpi_binary_file.h>

const int Dim = 3;
using comm_info_type = scfd::communication::mpi_comm_info;

using ordinal = std::int32_t;
using big_ordinal = std::ptrdiff_t;

using part_t = scfd::communication::rect_partitioner<Dim,ordinal,big_ordinal>;
using idx_t = scfd::static_vec::vec<ordinal, Dim>;
using big_idx_t = scfd::static_vec::vec<big_ordinal, Dim>;

using big_ord_vec_t = scfd::static_vec::vec<big_ordinal, Dim>;
using big_ord_rect_t = typename part_t::big_ord_rect_t;


struct shift //indexing temporal structure that returs indexes or seek position w.r.t. the first line.
{
    shift(const big_ord_vec_t &dom_size_p, const big_ord_rect_t& my_glob_rect_p):
    dom_size_(dom_size_p),
    my_own_glob_rect_(my_glob_rect_p)
    {
        Nx = dom_size_[0];
        Ny = dom_size_[1];
        Nz = dom_size_[2];
    }

    big_ordinal idx(const idx_t& i) const 
    {
        return ( (i(2)+my_own_glob_rect_.i1(2))*(Ny)*(Nx) + (i(1)+my_own_glob_rect_.i1(1))*(Nx) + (i(0)+my_own_glob_rect_.i1(0)) );
    }
    big_ordinal idx(const ordinal j, const ordinal k, const ordinal l) const
    {
        return idx({j,k,l});
    }

    big_ord_vec_t idx3(const idx_t& i) const
    {
        return {(i(0)+my_own_glob_rect_.i1(0)), (i(1)+my_own_glob_rect_.i1(1)), (i(2)+my_own_glob_rect_.i1(2)) };
    }

    big_ordinal shft(const idx_t& i) const
    {
        return ( (i(2)+my_own_glob_rect_.i1(2))*(Ny)*(Nx) + (i(1)+my_own_glob_rect_.i1(1))*(Nx) + (my_own_glob_rect_.i1(0)) );
    }   
    big_ordinal shft(const ordinal k, const ordinal l) const
    {
        return shft( {0, k, l} );
    }

    big_ord_rect_t my_own_glob_rect_;
    big_ord_vec_t dom_size_;
    big_ordinal Nx, Ny, Nz;
};



int main(int argc, char *argv[])
{

    scfd::communication::mpi_wrap mpi(argc, argv);
    comm_info_type comm_info = mpi.comm_world();
    auto nproc = comm_info.num_procs;
    auto myproc = comm_info.myid;
    
    int Nx = 30, Ny = 47, Nz = 29;
    part_t part(comm_info, {Nx, Ny, Nz});

    auto domain = part.get_dom_rect();
    auto my_loc_rect = part.get_own_loc_rect();
    auto loc_dom_sz = my_loc_rect.calc_size();
    big_ord_rect_t my_global_rect = part.get_own_rect();
    shift sft(domain.calc_size(), my_global_rect);
    shift sft_loc(loc_dom_sz, my_loc_rect);
    
    auto size_nd = my_global_rect.calc_size();
    auto n_loc = my_global_rect.calc_area();

    
    std::vector<double> v(n_loc, myproc);
    for(ordinal l=0;l<loc_dom_sz(2);l++)
    {
        for(ordinal k=0;k<loc_dom_sz(1);k++)
        {
            for(ordinal j=0;j<loc_dom_sz(0);j++)
            {
                auto big_idx = sft.idx3({j,k,l});
                double x = (static_cast<double>(big_idx(0)) - 0.5*Nx )/Nx;
                double y = (static_cast<double>(big_idx(1)) - 0.5*Ny )/Ny;
                double z = (static_cast<double>(big_idx(2)) - 0.5*Nz )/Nz;

                auto loc_idx = sft_loc.idx({j,k,l});
                v[loc_idx] = exp( 10.0*(-x*x-y*y-z*z) ); //filler function
            }
        }
    }
    std::vector<double> v_ref;
    std::copy(v.begin(), v.end(), std::back_inserter(v_ref));

    std::string file_name{"test_file_write.txt"};
    { //writing file
        scfd::communication::mpi_binary_file<double> file(comm_info, file_name);
        file.size( domain.calc_area() ); //instead of preallocate
        
        std::vector<double> buf(loc_dom_sz(0),0);
        for(ordinal l=0;l<loc_dom_sz(2);l++)
        {
            for(ordinal k=0;k<loc_dom_sz(1);k++)
            {
                #pragma omp parallel for
                for(ordinal j=0;j<loc_dom_sz(0);j++)
                {
                    auto loc_idx = sft_loc.idx({j,k,l});
                    buf[j] = v[loc_idx];
                }
                auto shft = sft.shft(k,l);
                // file.seek(shft);
                // file.write(buf.data(), loc_dom_sz(0));
                file.write_at( shft, buf.data(), loc_dom_sz(0) );
            }
        }
    }
    { //reading file
        scfd::communication::mpi_binary_file<double> file(comm_info, file_name, MPI_MODE_RDONLY);
        std::vector<double> buf(loc_dom_sz(0), 0);

        for(ordinal l=0;l<loc_dom_sz(2);l++)
        {
            for(ordinal k=0;k<loc_dom_sz(1);k++)
            {
                auto shft = sft.shft(k,l);
                file.seek(shft);
                file.read(loc_dom_sz(0), buf.data() );
                #pragma omp parallel for
                for(ordinal j=0;j<loc_dom_sz(0);j++)
                {
                    auto loc_idx = sft_loc.idx({j,k,l});
                    v[loc_idx] = buf[j];
                }               
            }
        } 
    }

    bool is_failed = false;
    for(ordinal j = 0; j<n_loc;j++)
    {
        if(std::abs(v_ref[j]-v[j]) > std::numeric_limits<double>::epsilon() )
        {
            std::cerr << "error at: " << j << " with ref = " << v_ref[j] << " and file read = "  << v[j] << std::endl;
            is_failed = true;
        }
    }
    if(is_failed)
    {
        std::cout << "FAILED" << std::endl;
    }
    else
    {
        std::cout << "PASSED" << std::endl;
    }

    return is_failed;
}