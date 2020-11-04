// Copyright © 2016-2020 Ryabkov Oleg Igorevich, Evstigneev Nikolay Mikhaylovitch

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

#ifndef __SCFD_UTILS_LOG_MSG_TYPE_H__
#define __SCFD_UTILS_LOG_MSG_TYPE_H__

namespace scfd
{
namespace utils
{

enum class log_msg_type 
{ 
    INFO, 
    /// INFO_ALL refers to multi-process applications (like MPI) and means message, 
    /// that must be said distintly by each process (like, 'i'm 1st; i'm second etc')
    INFO_ALL, 
    WARNING, 
    ERROR,
    DEBUG
};

}
}

#endif
