// Copyright © 2016,2017 Ryabkov Oleg Igorevich, Evstigneev Nikolay Mikhaylovitch

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

#ifndef __SCFD_UTILS_NESTED_EXCEPTION_TO_MULTISTRING_H__
#define __SCFD_UTILS_NESTED_EXCEPTION_TO_MULTISTRING_H__

#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace scfd
{
namespace utils
{

void print_nested_exception(std::ostream &os, const std::exception& e, int level =  0)
{
    os << std::string(2*level, ' ') << e.what();
    try
    {
        std::rethrow_if_nested(e);
    }
    catch (const std::exception& nested_exception)
    {
        os << std::endl;
        print_nested_exception(os, nested_exception, level + 1);
    }
    catch (...) 
    {
    }
}

std::string nested_exception_to_multistring(const std::exception& e)
{
    std::stringstream str_stream;
    print_nested_exception(str_stream, e);
    return str_stream.str();
}

}

}

#endif
