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

#ifndef __SCFD_ARRAYS_INDEX_SEQUENCE_H__
#define __SCFD_ARRAYS_INDEX_SEQUENCE_H__

namespace scfd
{
namespace arrays
{
namespace detail
{

template <class Ord, Ord... I>
class index_sequence {};

template <class Ord, Ord N, bool End, Ord ...I>
struct make_index_sequence_ : make_index_sequence_<Ord,N-1,N-1==0,N-1,I...> {};

template <class Ord, Ord N, Ord ...I>
struct make_index_sequence_<Ord,N,true,I...> : index_sequence<Ord,I...> {};

template <class Ord, Ord N, Ord ...I>
struct make_index_sequence : make_index_sequence_<Ord,N,N==0,I...> {};

}
}
}

#endif
