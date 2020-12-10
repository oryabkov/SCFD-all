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

#ifndef __SCFD_UTILS_INITCUDA_H__
#define __SCFD_UTILS_INITCUDA_H__

#include <cstdio>
#include <cuda_runtime.h>
#include <string>
#include "cuda_safe_call.h"
#include "log_std.h"

/// NOTE 'inline' here is used to allow multiply object files include

namespace scfd
{
namespace utils
{

/// If there is only only one CUDA device on system - arguments are ignored.
/// Otherwise, if any of the arguments is -1 then interactive choise is performed
/// Otherwise, if dev_num == -2, pci_id device is used (if exists)
/// Otherwise, if pci_id == -2, dev_num device is used (if exists)
/// Otherwise, exception is generated
/// On success returns used device number
/// Log must be satisfy LogCFormatted concept
/// NOTE log_lev,log order is taken because otherwise overload problems occurs (3 int parameters in line)
template<class Log>
inline int init_cuda(int log_lev, Log &log, int pci_id, int dev_num = -2)
{
    
    int count = 0;
    int i = 0;

    cudaGetDeviceCount(&count);
    if(count == 0)
    {
        throw std::runtime_error("init_cuda: There is no compartable device found\n");
    }
    
    int res_dev_num=0;    
    
    if (count>1)
    {
        if ((pci_id==-1)||(dev_num==-1))
        {
            for (i = 0; i < count; i++) 
            {
                cudaDeviceProp device_prop;
                cudaGetDeviceProperties(&device_prop, i);
                //printf( "#%i:   %s, pci-bus id:%i %i %i \n", i, (char*)&device_prop,device_prop.pciBusID,device_prop.pciDeviceID,device_prop.pciDomainID);
                log.info_f(log_lev, "init_cuda: #%i:   %s, pci-bus id:%i %i %i ", i, (char*)&device_prop,device_prop.pciBusID,device_prop.pciDeviceID,device_prop.pciDomainID);
            }            
            //printf("Device number for it to use>>>\n");
            log.info_f(log_lev, "init_cuda: Device number for it to use>>>\n");
            if (scanf("%i", &res_dev_num) != 1)
                throw std::runtime_error("init_cuda: problem with interactive device number input\n");
        }
        else if (pci_id==-2) 
        {
            res_dev_num = dev_num;
            cudaDeviceProp device_prop;
            cudaGetDeviceProperties(&device_prop, res_dev_num);

            //printf("Using #%i:   %s@[%i:%i:%i]\n",res_dev_num,(char*)&device_prop,device_prop.pciBusID,device_prop.pciDeviceID,device_prop.pciDomainID);
            log.info_f(log_lev, "init_cuda: Using #%i:   %s@[%i:%i:%i]",res_dev_num,(char*)&device_prop,device_prop.pciBusID,device_prop.pciDeviceID,device_prop.pciDomainID);
        }
        else if (dev_num==-2) 
        {
            cudaDeviceProp device_prop;
            bool found = false;
            for (int j=0;j<count;j++)
            {
                cudaGetDeviceProperties(&device_prop, j);
                if (device_prop.pciBusID==pci_id)
                {
                    res_dev_num = j;
                    found = true;
                    break;
                }
            }

            if (!found) 
                throw std::runtime_error("init_cuda: Did not find device with pci_id == " + std::to_string(pci_id));

            //printf("Using #%i:   %s@[%i:%i:%i]\n", res_dev_num,(char*)&device_prop,device_prop.pciBusID,device_prop.pciDeviceID,device_prop.pciDomainID);
            log.info_f(log_lev, "Using #%i:   %s@[%i:%i:%i]", res_dev_num,(char*)&device_prop,device_prop.pciBusID,device_prop.pciDeviceID,device_prop.pciDomainID);
        } 
        else 
        {
            throw std::runtime_error("init_cuda: Incorrect arguments: pci_id == " + std::to_string(pci_id) + 
                                     " dev_num == " + std::to_string(dev_num));
        }
    }
    else
    {
        cudaDeviceProp device_prop;
        cudaGetDeviceProperties(&device_prop, res_dev_num);
        /*printf("init_cuda: There is only one compartable CUDA device. It will be used regardless of preference.\n");
        printf( "#%i:   %s, pci-bus id:%i %i %i \n", res_dev_num, (char*)&device_prop,device_prop.pciBusID,device_prop.pciDeviceID,device_prop.pciDomainID);
        printf( "       using it...\n");*/
        log.info_f(log_lev, "init_cuda: There is only one compartable CUDA device. It will be used regardless of preference");
        log.info_f(log_lev, "init_cuda: #%i:   %s, pci-bus id:%i %i %i ", res_dev_num, (char*)&device_prop,device_prop.pciBusID,device_prop.pciDeviceID,device_prop.pciDomainID);
        log.info_f(log_lev, "init_cuda:        using it...");
    }

    CUDA_SAFE_CALL( cudaSetDevice(res_dev_num) );
    
    return res_dev_num;
}

/// The same as previous, but log_lev == 1
template<class Log>
inline int init_cuda(Log &log, int pci_id, int dev_num = -2)
{
    return init_cuda(1, log, pci_id, dev_num);
}

/// Default std::cout log version
// TODO make it without log_cformatted - create some 'direct' class for this
inline int init_cuda(int pci_id, int dev_num = -2)
{
    log_std   log;
    return init_cuda(log, pci_id, dev_num);
}

/// cuda_device_string:
/// either manual - for console interactive choise
/// either pci_id:XXX, (like pci_id:3)
/// either dev_num:XXX (like dev_num:0) 
template<class Log>
inline int init_cuda_str(Log &log, const std::string &cuda_device_string)
{
    std::string     value_str;
    int             pci_id = -2, dev_num = -2;
    if (cuda_device_string == "manual") 
    {
        pci_id = -1; dev_num = -1; 
    } 
    else if (cuda_device_string.substr(0, 7) == "pci_id:") 
    {
        value_str = cuda_device_string.substr(7);
        pci_id = std::stoi(value_str);
    } 
    else if (cuda_device_string.substr(0, 8) == "dev_num:") 
    {
        value_str = cuda_device_string.substr(8);
        dev_num = std::stoi(value_str);
    }
    else 
    {
        throw std::runtime_error("init_cuda_str: Argument " + cuda_device_string + " is incorrect");
    }

    return init_cuda(log, pci_id, dev_num);
}

inline int init_cuda_str(const std::string &cuda_device_string)
{
    log_std   log;
    return init_cuda_str(log, cuda_device_string);
}

inline std::string init_cuda_str_cmd_help(const std::string &arg_name)
{
    return arg_name + " argument sets up cuda device choice and could be:\n" + 
        "  either dev_num:DEVICE_NUMBER\n" + 
        "  either pci_id:PCI_ID\n" + 
        "  either manual for interactive console device choice";
}

}

}

#endif