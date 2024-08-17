#include <scfd/communication/mpi_wrap.h>
#include <scfd/communication/sequential_mpi_debug.h>
#include <iostream>



int main(int argc, char *argv[])
{
    scfd::communication::mpi_wrap mpi_wrap(argc, argv);
    auto mpi_comm = mpi_wrap.comm_world();
    scfd::communication::sequential_mpi_debug seq_mpi(mpi_comm);

    //this marks the section which will be executed in sequential order, from 0 to mpi_comm.num_procs
    if(mpi_comm.myid == 0)
    {
        std::cout << "the following output should be in sequential order of proc = ..., and proc_running = ..." << std::endl;
    }
    seq_mpi.start(); 
    auto proc_running = seq_mpi.get_rank();
    std::cout << "proc = " << mpi_comm.myid << ", proc_running = " << proc_running << std::endl;
    seq_mpi.stop();
    std::cout << "after_barrier: " << "proc = " << mpi_comm.myid << ", proc_running = " << proc_running << std::endl;


    return 0;
}