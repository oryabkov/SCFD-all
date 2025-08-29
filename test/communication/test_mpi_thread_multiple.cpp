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



#include <iostream>
#include <scfd/utils/log_mpi.h>
#include <scfd/communication/mpi_wrap.h>
#include <scfd/communication/mpi_comm_info.h>

using comm_info_type = scfd::communication::mpi_comm_info;

int main(int argc, char *argv[]) {
    
    scfd::communication::mpi_wrap mpi(argc, argv);
    comm_info_type comm_info = mpi.comm_world();
    int provided_threads = mpi.provided_threads();
    std::cout << "provided_threads: " << provided_threads << std::endl;

    return 0;
}