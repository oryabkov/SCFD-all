// Copyright © 2016 Ryabkov Oleg Igorevich, Evstigneev Nikolay Mikhaylovitch

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

#ifndef __SCFD_SERIAL_COMMUNICATOR_H__
#define __SCFD_SERIAL_COMMUNICATOR_H__

namespace scfd
{
namespace communication
{

class serial_communicator
{
public:
    int size()const
    {
        return 1;
    }
    int my_rank()const
    {
        return 0;
    }
    
    template<class T>
    T   reduce_max(const T &local_val)const
    {
        return local_val;
    }
    template<class T>
    T   reduce_sum(const T &local_val)const
    {
        return local_val;
    }
};

}  /// namespace communication
}  /// namespace scfd

#endif

