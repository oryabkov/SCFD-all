// Copyright © 2016-2018 Ryabkov Oleg Igorevich, Evstigneev Nikolay Mikhaylovitch

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

#ifndef __SCFD_UTILS_MAIN_TRY_CATCH_MACRO_H__
#define __SCFD_UTILS_MAIN_TRY_CATCH_MACRO_H__

#include <string>
#include <exception>
#include <chrono>
#include <ratio>
#include "log.h"

#ifndef SCFD_MAIN_TRY_CATCH_DISABLE_CATCH
#define SCFD_MAIN_TRY_CATCH_DISABLE_CATCH 0
#endif

#define USE_MAIN_TRY_CATCH(log_obj)                                                                     \
    auto                  *MAIN_TRY_CATCH_LOG_OBJ_POINTER = &log_obj;                                   \
    std::string           MAIN_TRY_CATCH_CURRENT_BLOCK_NAME;                                            \
    auto                  MAIN_TRY_CATCH_CURRENT_TIMER1 = std::chrono::high_resolution_clock::now();    \
    auto                  MAIN_TRY_CATCH_CURRENT_TIMER2 = std::chrono::high_resolution_clock::now();    \
    std::chrono::duration<double, std::milli> MAIN_TRY_CATCH_CURRENT_FP_MS;

#if SCFD_MAIN_TRY_CATCH_DISABLE_CATCH == 0

#define MAIN_TRY(block_name)                                                        \
    try {                                                                           \
        MAIN_TRY_CATCH_CURRENT_BLOCK_NAME = block_name;                             \
        MAIN_TRY_CATCH_LOG_OBJ_POINTER->info(block_name);                           \
        MAIN_TRY_CATCH_CURRENT_TIMER1 = std::chrono::high_resolution_clock::now();  

#define MAIN_CATCH(error_return_code)                                                                                                                         \
        MAIN_TRY_CATCH_CURRENT_TIMER2 = std::chrono::high_resolution_clock::now();                                                                            \
        MAIN_TRY_CATCH_CURRENT_FP_MS = MAIN_TRY_CATCH_CURRENT_TIMER2 - MAIN_TRY_CATCH_CURRENT_TIMER1;                                                         \
        MAIN_TRY_CATCH_LOG_OBJ_POINTER->info(MAIN_TRY_CATCH_CURRENT_BLOCK_NAME + " done");                                                                    \
        MAIN_TRY_CATCH_LOG_OBJ_POINTER->info_f("wall host time for " + MAIN_TRY_CATCH_CURRENT_BLOCK_NAME + ": %f ms", MAIN_TRY_CATCH_CURRENT_FP_MS.count());  \
    } catch (std::exception &e) {                                                                                                                             \
        MAIN_TRY_CATCH_LOG_OBJ_POINTER->error(std::string("error during ") + MAIN_TRY_CATCH_CURRENT_BLOCK_NAME + std::string(": ") + e.what());               \
        return error_return_code;                                                                                                                             \
    }

#else

#define MAIN_TRY(block_name)                                                    \
    MAIN_TRY_CATCH_CURRENT_BLOCK_NAME = block_name;                             \
    MAIN_TRY_CATCH_LOG_OBJ_POINTER->info(block_name);                           \
    MAIN_TRY_CATCH_CURRENT_TIMER1 = std::chrono::high_resolution_clock::now();  

#define MAIN_CATCH(error_return_code)                             
    MAIN_TRY_CATCH_CURRENT_TIMER2 = std::chrono::high_resolution_clock::now();                                                                            \
    MAIN_TRY_CATCH_CURRENT_FP_MS = MAIN_TRY_CATCH_CURRENT_TIMER2 - MAIN_TRY_CATCH_CURRENT_TIMER1;                                                         \
    MAIN_TRY_CATCH_LOG_OBJ_POINTER->info_f("wall host time for " + MAIN_TRY_CATCH_CURRENT_BLOCK_NAME + ": %f ms", MAIN_TRY_CATCH_CURRENT_FP_MS.count());  \

#endif

#endif
