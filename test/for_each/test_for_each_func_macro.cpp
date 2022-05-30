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


#include <cstdio>
#include <scfd/for_each/for_each_func_macro.h>

#define __STR2__(x) #x
#define __STR1__(x) __STR2__(x)

struct test
{
    //#pragma message("testtest!!" __STR1__( FOR_EACH_FUNC_NARG(1,1,2,2,3,3,4,4) ))
    //#pragma message("testtest!!" __STR1__( FOR_EACH_FUNC_PARAMS_HELP(test, int, a1, float, b1, int, a2, int, a3) ))
    FOR_EACH_FUNC_PARAMS_HELP(test, int, a1, float, b1, int, a2, int, a3)
};

int main()
{
    test    t(1,1.f,2,3);
    printf("%d %f %d %d\n", t.a1, t.b1, t.a2, t.a3);
    //t.a1 

}