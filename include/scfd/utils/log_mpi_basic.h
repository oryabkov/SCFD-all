// Copyright © 2016-2018 Ryabkov Oleg Igorevich, Evstigneev Nikolay Mikhaylovitch

// This file is part of SimpleCFD.

// SimpleCFD is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 2 only of the License.

// SimpleCFD is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with SimpleCFD.  If not, see <http://www.gnu.org/licenses/>.

#ifndef __SCFD_UTILS_LOG_MPI_BASIC_H__
#define __SCFD_UTILS_LOG_MPI_BASIC_H__

#include <string>
#include <exception>
#include <stdexcept>
#include <cstdarg>
#include <cstdio>
#include <mpi.h>
#include "log_msg_type.h"

namespace scfd
{
namespace utils
{

class log_mpi_basic
{
public:
    using log_msg_type = utils::log_msg_type;

private:
    int     log_lev;
    int     comm_rank_, comm_size_;

public:
    log_mpi_basic() : log_lev(1) 
    {  
        if (MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank_) != MPI_SUCCESS) throw std::runtime_error("log_mpi_basic::MPI_Comm_rank failed");
        if (MPI_Comm_size(MPI_COMM_WORLD, &comm_size_) != MPI_SUCCESS) throw std::runtime_error("log_mpi_basic::MPI_Comm_size failed");
    }

    void msg(const std::string &s, log_msg_type mt = log_msg_type::INFO, int _log_lev = 1)
    {
        if ((mt != log_msg_type::ERROR)&&(_log_lev > log_lev)) return;
        //TODO
        if (mt == log_msg_type::INFO) {
            if (comm_rank_ == 0) printf("INFO:         %s\n", s.c_str());
        } else if (mt == log_msg_type::INFO_ALL) {
            printf("INFO_ALL(%3d):%s\n", comm_rank_, s.c_str());
        } else if (mt == log_msg_type::WARNING) {
            printf("WARNING(%3d): %s\n", comm_rank_, s.c_str());
        } else if (mt == log_msg_type::ERROR) {
            printf("ERROR(%3d):   %s\n", comm_rank_, s.c_str());
        } else 
            throw std::logic_error("log_mpi_basic::log: wrong t_msg_type argument");
    }
    void set_verbosity(int _log_lev = 1) { log_lev = _log_lev; }

    int comm_rank()const
    {
        return comm_rank_;
    }
    int comm_size()const
    {
        return comm_size_;
    }
};

}

}

#endif
