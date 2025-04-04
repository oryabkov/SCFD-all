// Copyright © 2016-2025 Ryabkov Oleg Igorevich, Evstigneev Nikolay Mikhaylovitch, Sorokin Ivan Antonovich

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

#ifndef __SCFD_UTILS_INITHIP_H__
#define __SCFD_UTILS_INITHIP_H__

#include <cstdio>
#include <hip/hip_runtime.h>
#include <string>
#include <chrono>
#include <utility>
#include <tuple>
#include <vector>
#include <algorithm>
#include "hip_safe_call.h"
#include "log_std.h"

/// NOTE 'inline' here is used to allow multiply object files include

namespace scfd
{
namespace utils
{

/// If there is only only one HIP device on system - arguments are ignored.
/// Otherwise, if any of the arguments is -1 then interactive choise is performed
/// Otherwise, if dev_num == -2, pci_id device is used (if exists)
/// Otherwise, if pci_id == -2, dev_num device is used (if exists)
/// Otherwise, exception is generated
/// On success returns used device number
/// Log must be satisfy LogCFormatted concept
/// NOTE log_lev,log order is taken because otherwise overload problems occurs (3 int parameters in line)
template<class Log>
inline int init_hip(int log_lev, Log &log, int pci_id, int dev_num = -2)
{

    int count = 0;
    int i = 0;

    hipGetDeviceCount(&count);
    if(count == 0)
    {
        throw std::runtime_error("init_hip: There is no compartable device found\n");
    }

    int res_dev_num=0;

    if (count>1)
    {
        if ((pci_id==-1)||(dev_num==-1))
        {
            for (i = 0; i < count; i++)
            {
                hipDeviceProp_t device_prop;
                hipGetDeviceProperties(&device_prop, i);
                //printf( "#%i:   %s, pci-bus id:%i %i %i \n", i, (char*)&device_prop,device_prop.pciBusID,device_prop.pciDeviceID,device_prop.pciDomainID);
                log.info_f(log_lev, "init_hip: #%i:   %s, pci-bus id:%i %i %i ", i, (char*)&device_prop,device_prop.pciBusID,device_prop.pciDeviceID,device_prop.pciDomainID);
            }
            //printf("Device number for it to use>>>\n");
            log.info_f(log_lev, "init_hip: Device number for it to use>>>\n");
            if (scanf("%i", &res_dev_num) != 1)
                throw std::runtime_error("init_hip: problem with interactive device number input\n");
        }
        else if (pci_id==-2)
        {
            res_dev_num = dev_num;
            hipDeviceProp_t device_prop;
            hipGetDeviceProperties(&device_prop, res_dev_num);

            //printf("Using #%i:   %s@[%i:%i:%i]\n",res_dev_num,(char*)&device_prop,device_prop.pciBusID,device_prop.pciDeviceID,device_prop.pciDomainID);
            log.info_f(log_lev, "init_hip: Using #%i:   %s@[%i:%i:%i]",res_dev_num,(char*)&device_prop,device_prop.pciBusID,device_prop.pciDeviceID,device_prop.pciDomainID);
        }
        else if (dev_num==-2)
        {
            hipDeviceProp_t device_prop;
            bool found = false;
            for (int j=0;j<count;j++)
            {
                hipGetDeviceProperties(&device_prop, j);
                if (device_prop.pciBusID==pci_id)
                {
                    res_dev_num = j;
                    found = true;
                    break;
                }
            }

            if (!found)
                throw std::runtime_error("init_hip: Did not find device with pci_id == " + std::to_string(pci_id));

            //printf("Using #%i:   %s@[%i:%i:%i]\n", res_dev_num,(char*)&device_prop,device_prop.pciBusID,device_prop.pciDeviceID,device_prop.pciDomainID);
            log.info_f(log_lev, "Using #%i:   %s@[%i:%i:%i]", res_dev_num,(char*)&device_prop,device_prop.pciBusID,device_prop.pciDeviceID,device_prop.pciDomainID);
        }
        else
        {
            throw std::runtime_error("init_hip: Incorrect arguments: pci_id == " + std::to_string(pci_id) +
                                     " dev_num == " + std::to_string(dev_num));
        }
    }
    else
    {
        hipDeviceProp_t device_prop;
        hipGetDeviceProperties(&device_prop, res_dev_num);
        /*printf("init_hip: There is only one compartable HIP device. It will be used regardless of preference.\n");
        printf( "#%i:   %s, pci-bus id:%i %i %i \n", res_dev_num, (char*)&device_prop,device_prop.pciBusID,device_prop.pciDeviceID,device_prop.pciDomainID);
        printf( "       using it...\n");*/
        log.info_f(log_lev, "init_hip: There is only one compartable HIP device. It will be used regardless of preference");
        log.info_f(log_lev, "init_hip: #%i:   %s, pci-bus id:%i %i %i ", res_dev_num, (char*)&device_prop,device_prop.pciBusID,device_prop.pciDeviceID,device_prop.pciDomainID);
        log.info_f(log_lev, "init_hip:        using it...");
    }

    HIP_SAFE_CALL( hipSetDevice(res_dev_num) );

    return res_dev_num;
}

/// The same as previous, but log_lev == 1
template<class Log>
inline int init_hip(Log &log, int pci_id, int dev_num = -2)
{
    return init_hip(1, log, pci_id, dev_num);
}

/// Default std::cout log version
// TODO make it without log_cformatted - create some 'direct' class for this
inline int init_hip(int pci_id, int dev_num = -2)
{
    log_std   log;
    return init_hip(log, pci_id, dev_num);
}

/// hip_device_string:
/// either manual - for console interactive choise
/// either pci_id:XXX, (like pci_id:3)
/// either dev_num:XXX (like dev_num:0)
template<class Log>
inline int init_hip_str(Log &log, const std::string &hip_device_string)
{
    std::string     value_str;
    int             pci_id = -2, dev_num = -2;
    if (hip_device_string == "manual")
    {
        pci_id = -1; dev_num = -1;
    }
    else if (hip_device_string.substr(0, 7) == "pci_id:")
    {
        value_str = hip_device_string.substr(7);
        pci_id = std::stoi(value_str);
    }
    else if (hip_device_string.substr(0, 8) == "dev_num:")
    {
        value_str = hip_device_string.substr(8);
        dev_num = std::stoi(value_str);
    }
    else
    {
        throw std::runtime_error("init_hip_str: Argument " + hip_device_string + " is incorrect");
    }

    return init_hip(log, pci_id, dev_num);
}

inline int init_hip_str(const std::string &hip_device_string)
{
    log_std   log;
    return init_hip_str(log, hip_device_string);
}

// device_memory_in_MB = 0 <-automatic selection of a gpu with maximum free memory available
template<class Log>
inline int init_hip_persistent(Log &log, std::size_t device_memory_in_MB, std::size_t max_minutes = 10)
{
    int log_lev = 1;
    double save_coefficient = 0.98;

    auto start = std::chrono::steady_clock::now();

    bool device_is_set = false;
    int res_dev_num = 0;

    int count = 0;
    HIP_SAFE_CALL( hipGetDeviceCount(&count) );
    if(count == 0)
    {
        throw std::runtime_error("init_hip_persistent: There is no compartable device found\n");
    }
    using scft_init_hip_tuple_t = std::tuple<int,std::size_t, std::size_t>;
    std::vector<scft_init_hip_tuple_t> device_mems;

    std::size_t max_total_mem = 0;
    std::size_t max_free_mem = 0;

    for(int dev = 0;dev < count; dev++)
    {
        std::size_t free_mem_l;
        std::size_t total_mem_l;

        HIP_SAFE_CALL( hipSetDevice( dev ) );
        HIP_SAFE_CALL( hipMemGetInfo ( &free_mem_l, &total_mem_l ) );
        device_mems.push_back({dev, free_mem_l, total_mem_l});

        if( max_total_mem < total_mem_l )
        {
            max_total_mem = total_mem_l;
        }
        if( max_free_mem < free_mem_l )
        {
            max_free_mem = free_mem_l;
        }
        HIP_SAFE_CALL(hipDeviceReset());
    }
    if(device_memory_in_MB > 0)
    {
        if( std::size_t(save_coefficient*max_total_mem) < device_memory_in_MB*1000*1000)
        {
            throw std::runtime_error("init_hip_persistent: no suitable device found with minimum required memory.\n  Maximum found device memory is " + std::to_string(save_coefficient*max_total_mem/1000.0/1000.0) + "MB \n" );
        }
        while(true)
        {
            std::size_t free_mem, total_mem;
            for(int dev = 0;dev < count; dev++)
            {
                HIP_SAFE_CALL( hipSetDevice( dev ) );
                HIP_SAFE_CALL( hipMemGetInfo ( &free_mem, &total_mem ) );
                if (free_mem >= device_memory_in_MB*1000*1000)
                {
                    res_dev_num = dev;
                    device_is_set = true;
                    break;
                }
                HIP_SAFE_CALL( hipDeviceReset() );
            }
            if(device_is_set)
            {
                break;
            }



            auto current = std::chrono::steady_clock::now();
            auto current_duration = std::chrono::duration_cast<std::chrono::minutes>(current - start).count();
            if(current_duration >= max_minutes)
            {
                break;
            }

        }
    }
    else
    {
        std::sort(device_mems.begin(), device_mems.end(),
            []
            (const scft_init_hip_tuple_t& a, const scft_init_hip_tuple_t& b)
            {
                return std::get<1>(a) > std::get<1>(b);
            }
        );
        res_dev_num = std::get<0>(device_mems[0]);  //device with the largest ammount of free mem
        HIP_SAFE_CALL( hipSetDevice( res_dev_num ) );
        device_is_set = true;
    }


    if(!device_is_set)
    {
        throw std::runtime_error("init_hip_persistent: failed to find a suitable device for the given time interval.\n");
    }
    hipDeviceProp_t device_prop;
    HIP_SAFE_CALL( hipGetDeviceProperties(&device_prop, res_dev_num) );
    log.info_f(log_lev, "init_hip_persistent%s: Using #%i:   %s@[%i:%i:%i], %i MB", device_memory_in_MB==0?"_best_mem":" ", res_dev_num,(char*)&device_prop,device_prop.pciBusID, device_prop.pciDeviceID,device_prop.pciDomainID, std::size_t(device_prop.totalGlobalMem/1000.0/1000.0) );

    return res_dev_num;
}
// default is for automatic selection of a GPU with maximum available device memory
inline int init_hip_persistent(std::size_t device_memory = 0, std::size_t max_minutes = 10)
{
    log_std   log;
    return init_hip_persistent(log, device_memory, max_minutes);
}


inline std::string init_hip_str_cmd_help(const std::string &arg_name)
{
    return arg_name + " argument sets up hip device choice and could be:\n" +
        "  either dev_num:DEVICE_NUMBER\n" +
        "  either pci_id:PCI_ID\n" +
        "  either manual for interactive console device choice";
}

}

}

#endif
