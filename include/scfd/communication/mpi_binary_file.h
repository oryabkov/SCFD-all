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

#ifndef __SCFD_COMM_MPI_BINARY_FILE_H__
#define __SCFD_COMM_MPI_BINARY_FILE_H__

#include <string>
#include "mpi_comm_info.h"


namespace scfd
{
namespace communication
{

template<class T>
struct mpi_binary_file
{
    // • MPI_MODE_RDONLY — read only,
    // • MPI_MODE_RDWR — reading and writing,
    // • MPI_MODE_WRONLY — write only,
    // • MPI_MODE_CREATE — create the file if it does not exist,
    // • MPI_MODE_EXCL — error if creating file that already exists,
    // • MPI_MODE_DELETE_ON_CLOSE — delete file on close,
    // • MPI_MODE_UNIQUE_OPEN — file will not be concurrently opened elsewhere,
    // • MPI_MODE_SEQUENTIAL — file will only be accessed sequentially,
    // • MPI_MODE_APPEND — set initial position of all file pointers to end of file.    
    mpi_binary_file(const mpi_comm_info& comm_info, const std::string& file_name, int amode = MPI_MODE_RDWR|MPI_MODE_CREATE)
    {
        SCFD_MPI_SAFE_CALL(MPI_File_open(comm_info.comm, file_name.c_str(), amode, MPI_INFO_NULL, &fh));
    }
    ~mpi_binary_file()
    {
        MPI_File_close(&fh);
    }
    // sizeof(MPI_OFFSET) = 8
    // • MPI_SEEK_SET: the pointer is set to offset
    // • MPI_SEEK_CUR: the pointer is set to the current pointer position plus offset
    // • MPI_SEEK_END: the pointer is set to the end of file plus offset    
    void seek(MPI_Offset offset, int whence = MPI_SEEK_SET) const
    {
        SCFD_MPI_SAFE_CALL( MPI_File_seek(fh, offset*sizeof(T), whence) );
    }
    void size(MPI_Offset size) const
    {
        SCFD_MPI_SAFE_CALL( MPI_File_set_size(fh, size*sizeof(T)) );
    }
    void preallocate(MPI_Offset size) const //WARNING! seems to make file larger if repeatedly applied to the same existing file. Use 'size()' insted.
    {
        SCFD_MPI_SAFE_CALL( MPI_File_preallocate(fh, size*sizeof(T) ) );
    }
    void write(const T* data, int count) const
    {
        SCFD_MPI_SAFE_CALL( MPI_File_write(fh, data, count*sizeof(T), MPI_BYTE, MPI_STATUS_IGNORE) );
    }
    void write_at(MPI_Offset offset, const T* data, int count) const
    {
        SCFD_MPI_SAFE_CALL( MPI_File_write_at(fh, offset*sizeof(T), data, count*sizeof(T), MPI_BYTE, MPI_STATUS_IGNORE) );
    }
    void read(int count, T* data) const
    {    
        SCFD_MPI_SAFE_CALL( MPI_File_read(fh, data, count*sizeof(T), MPI_BYTE, MPI_STATUS_IGNORE) );
    }
    void read_at(MPI_Offset offset, int count, T* data) const //WARNING! for some reason it doesn't work as expexted!
    {    
        SCFD_MPI_SAFE_CALL( MPI_File_read_at(fh, offset*sizeof(T), data, count, MPI_BYTE, MPI_STATUS_IGNORE) );
    }

    MPI_File& handle()
    {
        return fh;
    }

private:
    MPI_File fh;

};


} // namespace communication
} // namespace scfd

#endif