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

#ifndef __SCFD_FOR_EACH_FUNC_MACRO_H__
#define __SCFD_FOR_EACH_FUNC_MACRO_H__

#include "for_each_config.h"

//TODO not working for MSVC 2008
//i suppose we can use boost instead?
//NOTE however this variant is totally standart-compliant

#define SCFD_FOR_EACH_FUNC_CONCATENATE(arg1, arg2)   SCFD_FOR_EACH_FUNC_CONCATENATE1(arg1, arg2)
#define SCFD_FOR_EACH_FUNC_CONCATENATE1(arg1, arg2)  SCFD_FOR_EACH_FUNC_CONCATENATE2(arg1, arg2)
#define SCFD_FOR_EACH_FUNC_CONCATENATE2(arg1, arg2)  arg1##arg2

#define SCFD_FOR_EACH_FUNC_NARG(...) SCFD_FOR_EACH_FUNC_NARG_(__VA_ARGS__, SCFD_FOR_EACH_FUNC_RSEQ_N())
#define SCFD_FOR_EACH_FUNC_NARG_(...) SCFD_FOR_EACH_FUNC_ARG_N(__VA_ARGS__) 
#define SCFD_FOR_EACH_FUNC_ARG_N(_11, _12, _21, _22, _31, _32, _41, _42, _51, _52, _61, _62, _71, _72, _81, _82, _91, _92, _101, _102, _111, _112, _121, _122, _131, _132, _141, _142, _151, _152, _161, _162, _171, _172, _181, _182, N, ...) N 
#define SCFD_FOR_EACH_FUNC_RSEQ_N() 18, 18, 17, 17, 16, 16, 15, 15, 14, 14, 13, 13, 12, 12, 11, 11, 10, 10, 9, 9, 8, 8, 7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0

#define SCFD_FOR_EACH_FUNC_PARAMS_DEF_1(par_type, par_name, ...) par_type par_name;
#define SCFD_FOR_EACH_FUNC_PARAMS_DEF_2(par_type, par_name, ...)                             \
  par_type par_name;                                                                    \
  SCFD_FOR_EACH_FUNC_PARAMS_DEF_1(__VA_ARGS__)
#define SCFD_FOR_EACH_FUNC_PARAMS_DEF_3(par_type, par_name, ...)                             \
  par_type par_name;                                                                    \
  SCFD_FOR_EACH_FUNC_PARAMS_DEF_2(__VA_ARGS__)
#define SCFD_FOR_EACH_FUNC_PARAMS_DEF_4(par_type, par_name, ...)                             \
  par_type par_name;                                                                    \
  SCFD_FOR_EACH_FUNC_PARAMS_DEF_3(__VA_ARGS__)
#define SCFD_FOR_EACH_FUNC_PARAMS_DEF_5(par_type, par_name, ...)                             \
  par_type par_name;                                                                    \
  SCFD_FOR_EACH_FUNC_PARAMS_DEF_4(__VA_ARGS__)
#define SCFD_FOR_EACH_FUNC_PARAMS_DEF_6(par_type, par_name, ...)                             \
  par_type par_name;                                                                    \
  SCFD_FOR_EACH_FUNC_PARAMS_DEF_5(__VA_ARGS__)
#define SCFD_FOR_EACH_FUNC_PARAMS_DEF_7(par_type, par_name, ...)                             \
  par_type par_name;                                                                    \
  SCFD_FOR_EACH_FUNC_PARAMS_DEF_6(__VA_ARGS__)
#define SCFD_FOR_EACH_FUNC_PARAMS_DEF_8(par_type, par_name, ...)                             \
  par_type par_name;                                                                    \
  SCFD_FOR_EACH_FUNC_PARAMS_DEF_7(__VA_ARGS__)
#define SCFD_FOR_EACH_FUNC_PARAMS_DEF_9(par_type, par_name, ...)                             \
  par_type par_name;                                                                    \
  SCFD_FOR_EACH_FUNC_PARAMS_DEF_8(__VA_ARGS__)
#define SCFD_FOR_EACH_FUNC_PARAMS_DEF_10(par_type, par_name, ...)                            \
  par_type par_name;                                                                    \
  SCFD_FOR_EACH_FUNC_PARAMS_DEF_9(__VA_ARGS__)
#define SCFD_FOR_EACH_FUNC_PARAMS_DEF_11(par_type, par_name, ...)                            \
  par_type par_name;                                                                    \
  SCFD_FOR_EACH_FUNC_PARAMS_DEF_10(__VA_ARGS__)
#define SCFD_FOR_EACH_FUNC_PARAMS_DEF_12(par_type, par_name, ...)                            \
  par_type par_name;                                                                    \
  SCFD_FOR_EACH_FUNC_PARAMS_DEF_11(__VA_ARGS__)
#define SCFD_FOR_EACH_FUNC_PARAMS_DEF_13(par_type, par_name, ...)                            \
  par_type par_name;                                                                    \
  SCFD_FOR_EACH_FUNC_PARAMS_DEF_12(__VA_ARGS__)
#define SCFD_FOR_EACH_FUNC_PARAMS_DEF_14(par_type, par_name, ...)                            \
  par_type par_name;                                                                    \
  SCFD_FOR_EACH_FUNC_PARAMS_DEF_13(__VA_ARGS__)
#define SCFD_FOR_EACH_FUNC_PARAMS_DEF_15(par_type, par_name, ...)                            \
  par_type par_name;                                                                    \
  SCFD_FOR_EACH_FUNC_PARAMS_DEF_14(__VA_ARGS__)
#define SCFD_FOR_EACH_FUNC_PARAMS_DEF_16(par_type, par_name, ...)                            \
  par_type par_name;                                                                    \
  SCFD_FOR_EACH_FUNC_PARAMS_DEF_15(__VA_ARGS__)
#define SCFD_FOR_EACH_FUNC_PARAMS_DEF_17(par_type, par_name, ...)                            \
  par_type par_name;                                                                    \
  SCFD_FOR_EACH_FUNC_PARAMS_DEF_16(__VA_ARGS__)
#define SCFD_FOR_EACH_FUNC_PARAMS_DEF_18(par_type, par_name, ...)                            \
  par_type par_name;                                                                    \
  SCFD_FOR_EACH_FUNC_PARAMS_DEF_17(__VA_ARGS__)

#define SCFD_FOR_EACH_FUNC_PARAMS_DEF_(N, ...) SCFD_FOR_EACH_FUNC_CONCATENATE(SCFD_FOR_EACH_FUNC_PARAMS_DEF_, N)(__VA_ARGS__)
#define SCFD_FOR_EACH_FUNC_PARAMS_DEF(...) SCFD_FOR_EACH_FUNC_PARAMS_DEF_(SCFD_FOR_EACH_FUNC_NARG(__VA_ARGS__), __VA_ARGS__)

#define SCFD_FOR_EACH_FUNC_PARAMS_LIST_1(par_type, par_name, ...) par_type SCFD_FOR_EACH_FUNC_CONCATENATE(_,par_name)
#define SCFD_FOR_EACH_FUNC_PARAMS_LIST_2(par_type, par_name, ...)                                                            \
  par_type SCFD_FOR_EACH_FUNC_CONCATENATE(_,par_name),                                                                       \
  SCFD_FOR_EACH_FUNC_PARAMS_LIST_1(__VA_ARGS__)
#define SCFD_FOR_EACH_FUNC_PARAMS_LIST_3(par_type, par_name, ...)                                                            \
  par_type SCFD_FOR_EACH_FUNC_CONCATENATE(_,par_name),                                                                       \
  SCFD_FOR_EACH_FUNC_PARAMS_LIST_2(__VA_ARGS__)
#define SCFD_FOR_EACH_FUNC_PARAMS_LIST_4(par_type, par_name, ...)                                                            \
  par_type SCFD_FOR_EACH_FUNC_CONCATENATE(_,par_name),                                                                       \
  SCFD_FOR_EACH_FUNC_PARAMS_LIST_3(__VA_ARGS__)
#define SCFD_FOR_EACH_FUNC_PARAMS_LIST_5(par_type, par_name, ...)                                                            \
  par_type SCFD_FOR_EACH_FUNC_CONCATENATE(_,par_name),                                                                       \
  SCFD_FOR_EACH_FUNC_PARAMS_LIST_4(__VA_ARGS__)
#define SCFD_FOR_EACH_FUNC_PARAMS_LIST_6(par_type, par_name, ...)                                                            \
  par_type SCFD_FOR_EACH_FUNC_CONCATENATE(_,par_name),                                                                       \
  SCFD_FOR_EACH_FUNC_PARAMS_LIST_5(__VA_ARGS__)
#define SCFD_FOR_EACH_FUNC_PARAMS_LIST_7(par_type, par_name, ...)                                                            \
  par_type SCFD_FOR_EACH_FUNC_CONCATENATE(_,par_name),                                                                       \
  SCFD_FOR_EACH_FUNC_PARAMS_LIST_6(__VA_ARGS__)
#define SCFD_FOR_EACH_FUNC_PARAMS_LIST_8(par_type, par_name, ...)                                                            \
  par_type SCFD_FOR_EACH_FUNC_CONCATENATE(_,par_name),                                                                       \
  SCFD_FOR_EACH_FUNC_PARAMS_LIST_7(__VA_ARGS__)
#define SCFD_FOR_EACH_FUNC_PARAMS_LIST_9(par_type, par_name, ...)                                                            \
  par_type SCFD_FOR_EACH_FUNC_CONCATENATE(_,par_name),                                                                       \
  SCFD_FOR_EACH_FUNC_PARAMS_LIST_8(__VA_ARGS__)
#define SCFD_FOR_EACH_FUNC_PARAMS_LIST_10(par_type, par_name, ...)                                                           \
  par_type SCFD_FOR_EACH_FUNC_CONCATENATE(_,par_name),                                                                       \
  SCFD_FOR_EACH_FUNC_PARAMS_LIST_9(__VA_ARGS__)
#define SCFD_FOR_EACH_FUNC_PARAMS_LIST_11(par_type, par_name, ...)                                                           \
  par_type SCFD_FOR_EACH_FUNC_CONCATENATE(_,par_name),                                                                       \
  SCFD_FOR_EACH_FUNC_PARAMS_LIST_10(__VA_ARGS__)
#define SCFD_FOR_EACH_FUNC_PARAMS_LIST_12(par_type, par_name, ...)                                                           \
  par_type SCFD_FOR_EACH_FUNC_CONCATENATE(_,par_name),                                                                       \
  SCFD_FOR_EACH_FUNC_PARAMS_LIST_11(__VA_ARGS__)
#define SCFD_FOR_EACH_FUNC_PARAMS_LIST_13(par_type, par_name, ...)                                                           \
  par_type SCFD_FOR_EACH_FUNC_CONCATENATE(_,par_name),                                                                       \
  SCFD_FOR_EACH_FUNC_PARAMS_LIST_12(__VA_ARGS__)
#define SCFD_FOR_EACH_FUNC_PARAMS_LIST_14(par_type, par_name, ...)                                                           \
  par_type SCFD_FOR_EACH_FUNC_CONCATENATE(_,par_name),                                                                       \
  SCFD_FOR_EACH_FUNC_PARAMS_LIST_13(__VA_ARGS__)
#define SCFD_FOR_EACH_FUNC_PARAMS_LIST_15(par_type, par_name, ...)                                                           \
  par_type SCFD_FOR_EACH_FUNC_CONCATENATE(_,par_name),                                                                       \
  SCFD_FOR_EACH_FUNC_PARAMS_LIST_14(__VA_ARGS__)
#define SCFD_FOR_EACH_FUNC_PARAMS_LIST_16(par_type, par_name, ...)                                                           \
  par_type SCFD_FOR_EACH_FUNC_CONCATENATE(_,par_name),                                                                       \
  SCFD_FOR_EACH_FUNC_PARAMS_LIST_15(__VA_ARGS__)
#define SCFD_FOR_EACH_FUNC_PARAMS_LIST_17(par_type, par_name, ...)                                                           \
  par_type SCFD_FOR_EACH_FUNC_CONCATENATE(_,par_name),                                                                       \
  SCFD_FOR_EACH_FUNC_PARAMS_LIST_16(__VA_ARGS__)
#define SCFD_FOR_EACH_FUNC_PARAMS_LIST_18(par_type, par_name, ...)                                                           \
  par_type SCFD_FOR_EACH_FUNC_CONCATENATE(_,par_name),                                                                       \
  SCFD_FOR_EACH_FUNC_PARAMS_LIST_17(__VA_ARGS__)

#define SCFD_FOR_EACH_FUNC_PARAMS_LIST_(N, ...) SCFD_FOR_EACH_FUNC_CONCATENATE(SCFD_FOR_EACH_FUNC_PARAMS_LIST_, N)(__VA_ARGS__)
#define SCFD_FOR_EACH_FUNC_PARAMS_LIST(...) SCFD_FOR_EACH_FUNC_PARAMS_LIST_(SCFD_FOR_EACH_FUNC_NARG(__VA_ARGS__), __VA_ARGS__)

#define SCFD_FOR_EACH_FUNC_PARAMS_CC_1(par_type, par_name, ...) par_name(SCFD_FOR_EACH_FUNC_CONCATENATE(_,par_name))
#define SCFD_FOR_EACH_FUNC_PARAMS_CC_2(par_type, par_name, ...)                                                              \
  par_name(SCFD_FOR_EACH_FUNC_CONCATENATE(_,par_name)),                                                                      \
  SCFD_FOR_EACH_FUNC_PARAMS_CC_1(__VA_ARGS__)
#define SCFD_FOR_EACH_FUNC_PARAMS_CC_3(par_type, par_name, ...)                                                              \
  par_name(SCFD_FOR_EACH_FUNC_CONCATENATE(_,par_name)),                                                                      \
  SCFD_FOR_EACH_FUNC_PARAMS_CC_2(__VA_ARGS__)
#define SCFD_FOR_EACH_FUNC_PARAMS_CC_4(par_type, par_name, ...)                                                              \
  par_name(SCFD_FOR_EACH_FUNC_CONCATENATE(_,par_name)),                                                                      \
  SCFD_FOR_EACH_FUNC_PARAMS_CC_3(__VA_ARGS__)
#define SCFD_FOR_EACH_FUNC_PARAMS_CC_5(par_type, par_name, ...)                                                              \
  par_name(SCFD_FOR_EACH_FUNC_CONCATENATE(_,par_name)),                                                                      \
  SCFD_FOR_EACH_FUNC_PARAMS_CC_4(__VA_ARGS__)
#define SCFD_FOR_EACH_FUNC_PARAMS_CC_6(par_type, par_name, ...)                                                              \
  par_name(SCFD_FOR_EACH_FUNC_CONCATENATE(_,par_name)),                                                                      \
  SCFD_FOR_EACH_FUNC_PARAMS_CC_5(__VA_ARGS__)
#define SCFD_FOR_EACH_FUNC_PARAMS_CC_7(par_type, par_name, ...)                                                              \
  par_name(SCFD_FOR_EACH_FUNC_CONCATENATE(_,par_name)),                                                                      \
  SCFD_FOR_EACH_FUNC_PARAMS_CC_6(__VA_ARGS__)
#define SCFD_FOR_EACH_FUNC_PARAMS_CC_8(par_type, par_name, ...)                                                              \
  par_name(SCFD_FOR_EACH_FUNC_CONCATENATE(_,par_name)),                                                                      \
  SCFD_FOR_EACH_FUNC_PARAMS_CC_7(__VA_ARGS__)
#define SCFD_FOR_EACH_FUNC_PARAMS_CC_9(par_type, par_name, ...)                                                              \
  par_name(SCFD_FOR_EACH_FUNC_CONCATENATE(_,par_name)),                                                                      \
  SCFD_FOR_EACH_FUNC_PARAMS_CC_8(__VA_ARGS__)
#define SCFD_FOR_EACH_FUNC_PARAMS_CC_10(par_type, par_name, ...)                                                             \
  par_name(SCFD_FOR_EACH_FUNC_CONCATENATE(_,par_name)),                                                                      \
  SCFD_FOR_EACH_FUNC_PARAMS_CC_9(__VA_ARGS__)
#define SCFD_FOR_EACH_FUNC_PARAMS_CC_11(par_type, par_name, ...)                                                             \
  par_name(SCFD_FOR_EACH_FUNC_CONCATENATE(_,par_name)),                                                                      \
  SCFD_FOR_EACH_FUNC_PARAMS_CC_10(__VA_ARGS__)
#define SCFD_FOR_EACH_FUNC_PARAMS_CC_12(par_type, par_name, ...)                                                             \
  par_name(SCFD_FOR_EACH_FUNC_CONCATENATE(_,par_name)),                                                                      \
  SCFD_FOR_EACH_FUNC_PARAMS_CC_11(__VA_ARGS__)
#define SCFD_FOR_EACH_FUNC_PARAMS_CC_13(par_type, par_name, ...)                                                             \
  par_name(SCFD_FOR_EACH_FUNC_CONCATENATE(_,par_name)),                                                                      \
  SCFD_FOR_EACH_FUNC_PARAMS_CC_12(__VA_ARGS__)
#define SCFD_FOR_EACH_FUNC_PARAMS_CC_14(par_type, par_name, ...)                                                             \
  par_name(SCFD_FOR_EACH_FUNC_CONCATENATE(_,par_name)),                                                                      \
  SCFD_FOR_EACH_FUNC_PARAMS_CC_13(__VA_ARGS__)
#define SCFD_FOR_EACH_FUNC_PARAMS_CC_15(par_type, par_name, ...)                                                             \
  par_name(SCFD_FOR_EACH_FUNC_CONCATENATE(_,par_name)),                                                                      \
  SCFD_FOR_EACH_FUNC_PARAMS_CC_14(__VA_ARGS__)
#define SCFD_FOR_EACH_FUNC_PARAMS_CC_16(par_type, par_name, ...)                                                             \
  par_name(SCFD_FOR_EACH_FUNC_CONCATENATE(_,par_name)),                                                                      \
  SCFD_FOR_EACH_FUNC_PARAMS_CC_15(__VA_ARGS__)
#define SCFD_FOR_EACH_FUNC_PARAMS_CC_17(par_type, par_name, ...)                                                             \
  par_name(SCFD_FOR_EACH_FUNC_CONCATENATE(_,par_name)),                                                                      \
  SCFD_FOR_EACH_FUNC_PARAMS_CC_16(__VA_ARGS__)
#define SCFD_FOR_EACH_FUNC_PARAMS_CC_18(par_type, par_name, ...)                                                             \
  par_name(SCFD_FOR_EACH_FUNC_CONCATENATE(_,par_name)),                                                                      \
  SCFD_FOR_EACH_FUNC_PARAMS_CC_17(__VA_ARGS__)

#define SCFD_FOR_EACH_FUNC_PARAMS_CC_(N, ...) SCFD_FOR_EACH_FUNC_CONCATENATE(SCFD_FOR_EACH_FUNC_PARAMS_CC_, N)(__VA_ARGS__)
#define SCFD_FOR_EACH_FUNC_PARAMS_CC(...) SCFD_FOR_EACH_FUNC_PARAMS_CC_(SCFD_FOR_EACH_FUNC_NARG(__VA_ARGS__), __VA_ARGS__)

// TODO this name is left for compability - don't use and remove in future
#define FOR_EACH_FUNC_PARAMS_HELP(func_name, ...)                                                                       \
        SCFD_FOR_EACH_FUNC_PARAMS_DEF(__VA_ARGS__)                                                                           \
        func_name(SCFD_FOR_EACH_FUNC_PARAMS_LIST(__VA_ARGS__)) : SCFD_FOR_EACH_FUNC_PARAMS_CC(__VA_ARGS__) {}     

#define SCFD_FOR_EACH_FUNC_PARAMS(func_name, ...)                                                                       \
        SCFD_FOR_EACH_FUNC_PARAMS_DEF(__VA_ARGS__)                                                                           \
        func_name(SCFD_FOR_EACH_FUNC_PARAMS_LIST(__VA_ARGS__)) : SCFD_FOR_EACH_FUNC_PARAMS_CC(__VA_ARGS__) {}

#endif
