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

#ifndef __SCFD_ARRAYS_HAS_SUBSCRIPT_OPERATOR_H__
#define __SCFD_ARRAYS_HAS_SUBSCRIPT_OPERATOR_H__

namespace scfd
{
namespace arrays
{
namespace detail
{

template<class T, class Index>
struct has_subscript_operator_impl
{
  template<class T1,
           class IndexDeduced = Index,
           class Reference = decltype(
             (*std::declval<T*>())[std::declval<IndexDeduced>()]
           ),
           class = typename std::enable_if<
             !std::is_void<Reference>::value
           >::type>
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<T>(0));
};

template<class T, class Index>
using has_subscript_operator = typename has_subscript_operator_impl<T,Index>::type;

}
}
}

#endif
