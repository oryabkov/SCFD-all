#ifndef __SCFD_UTILS_FORMAT_JSON_H__
#define __SCFD_UTILS_FORMAT_JSON_H__

#include <iostream>
#include <nlohmann/json.hpp>

namespace scfd
{
namespace utils
{

namespace detail
{

void print_ident(std::ostream &s, int level)
{
    s << "|";
    for (int i = 0;i < level;++i)
    {
        s << "|  ";
    }

}

} // namespace detail

void format_json(const nlohmann::json& j, std::ostream &s, int level = 0)
{
    for (auto& e : j.items())
    {
        if (e.value().is_object()) continue;
        detail::print_ident(s, level);
        s << "|==" << e.key() << ": " << e.value() << '\n';
    }
    for (auto& e : j.items())
    {
        if (!e.value().is_object()) continue;
        detail::print_ident(s, level);
        s << "|" << e.key() << ":\n";
        format_json(e.value(), s, level+1);
    }
}

} // namespace utils
} // namespace scfd

#endif
