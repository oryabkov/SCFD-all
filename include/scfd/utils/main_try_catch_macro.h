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
#include "log.h"

#define USE_MAIN_TRY_CATCH(log_obj)                                       \
    auto                  *MAIN_TRY_CATCH_LOG_OBJ_POINTER = &log_obj;     \
    std::string           MAIN_TRY_CATCH_CURRENT_BLOCK_NAME;

#define MAIN_TRY(block_name) try {                                  \
    MAIN_TRY_CATCH_CURRENT_BLOCK_NAME = block_name;                 \
    MAIN_TRY_CATCH_LOG_OBJ_POINTER->info(block_name);

#define MAIN_CATCH(error_return_code) } catch (std::exception &e) {                                                                                     \
        MAIN_TRY_CATCH_LOG_OBJ_POINTER->error(std::string("error during ") + MAIN_TRY_CATCH_CURRENT_BLOCK_NAME + std::string(": ") + e.what());         \
        return error_return_code;                                                                                                                       \
    }

#endif
