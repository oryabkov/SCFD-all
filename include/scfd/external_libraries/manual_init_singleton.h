#ifndef __SCFD_MANUAL_INIT_SINGLETON_H__
#define __SCFD_MANUAL_INIT_SINGLETON_H__

#include <stdexcept>

namespace scfd
{

template<class DerivedClass>
class manual_init_singleton
{
public:
    manual_init_singleton() = default;
    manual_init_singleton(DerivedClass *inst, bool do_set_inst = false)
    {
        if (do_set_inst)
        {
            set_inst(inst);
        }
    }

    static void set_inst(DerivedClass *inst)
    {
        inst_ = inst;
    }
    static DerivedClass &inst()
    {
        if (inst_ == nullptr) throw std::logic_error("manual_init_singleton::inst(): sigleton is not inited");
        return *inst_;
    }

protected:
    static DerivedClass *inst_;
};

}

#endif
