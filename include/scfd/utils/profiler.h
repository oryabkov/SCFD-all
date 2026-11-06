#ifndef __SCFD_UTILS_PROFILER_H__
#define __SCFD_UTILS_PROFILER_H__

/// Redone from Demidovs amgcl profiler with Counter subst with scfd events
/// and other minor changes
/// Initial copyrights:
/*
The MIT License

Copyright (c) 2012-2021 Denis Demidov <dennis.demidov@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/**
 * \file   amgcl/profiler.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  Profiler class.
 */

#include <iostream>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <type_traits>

#include <scfd/utils/manual_init_singleton.h>
#include <scfd/utils/system_timer_event.h>

namespace scfd
{

namespace utils
{

/// Profiler class.
/**
 * \param Event       One of the timer events classes to measure events points
 * \param SHIFT_WIDTH Indentation for output of profiling results.
 *
 * Provides simple to use, hierarchical profile with nicely formatted output.
 */
template <class Event = system_timer_event, unsigned SHIFT_WIDTH = 2>
class profiler : public manual_init_singleton<profiler<Event,SHIFT_WIDTH>>
{
    public:
        typedef double delta_type;

        /// Initialization.
        /**
         */
        profiler() : name("Profile")
        {
            init();
        }

        /// Sets name
        /**
         * \param name Profile title to use with output.
         */
        profiler(const std::string &name)
            : name(name)
        {
            init();
        }

        /// Starts measurement.
        /**
         * \param name interval name.
         */
        void tic(const std::string &name)
        {
            //stack.back()->children[name].begin = counter.current();
            if (stack.back()->children.find(name) == stack.back()->children.end())
            {
                std::size_t next_call_index = stack.back()->children.size();
                stack.back()->children[name].call_index = next_call_index;
            }
            stack.back()->children[name].begin_event.record();
            stack.push_back(&stack.back()->children[name]);
        }

        /// Stops measurement.
        /**
         * Returns delta in the measured value since the corresponding tic().
         */
        delta_type toc(const std::string& /*name*/ = "")
        {
            profile_unit *top = stack.back();
            stack.pop_back();

            Event       current_event;
            current_event.record();
            delta_type  delta   = current_event.elapsed_time(top->begin_event);

            top->length += delta;
            root.length = current_event.elapsed_time(root.begin_event);

            return delta;
        }

        void reset()
        {
            stack.clear();
            root.length = 0;
            root.children.clear();

            stack.push_back(&root);
            //root.begin = counter.current();
            root.begin_event.record();
        }

        struct scoped_ticker
        {
            profiler &prof;
            scoped_ticker(profiler &prof) : prof(prof) {}
            ~scoped_ticker() {
                prof.toc();
            }
        };

        scoped_ticker scoped_tic(const std::string &name)
        {
            tic(name);
            return scoped_ticker(*this);
        }

        template<class Log>
        struct scoped_ticker_print
        {
            profiler &prof;
            Log *log;
            std::string name;
            scoped_ticker_print(profiler &prof, Log *log, const std::string &name) : prof(prof), log(log), name(name) {}
            ~scoped_ticker_print() {
                delta_type delta = prof.toc();
                if (log != nullptr)
                {
                    log->info_f("%s: %.3f %s", name.c_str(), delta, Event::units().c_str());
                }
            }
        };

        template<class Log>
        scoped_ticker_print<Log> scoped_tic_print(const std::string &name, Log *log)
        {
            tic(name);
            return scoped_ticker_print<Log>(*this, log, name);
        }

        template<class Log>
        void log_print(Log &log)
        {
            std::stringstream sstream;
            print(sstream);
            log.info(sstream.str());
        }
        template<class Log>
        void log_print_totals(Log &log)
        {
            std::stringstream sstream;
            print_totals(sstream);
            log.info(sstream.str());
        }
    private:
        struct profile_unit
        {
            profile_unit() : length(0),call_index(0) {}

            delta_type children_time() const
            {
                delta_type s = delta_type();
                for(typename std::map<std::string, profile_unit>::const_iterator c = children.begin(); c != children.end(); c++)
                    s += c->second.length;
                return s;
            }

            size_t total_width(const std::string &name, int level) const
            {
                size_t w = name.size() + level;
                for(typename std::map<std::string, profile_unit>::const_iterator c = children.begin(); c != children.end(); c++)
                    w = std::max(w, c->second.total_width(c->first, level + SHIFT_WIDTH));
                return w;
            }

            void print(std::ostream &out, const std::string &name,
                    int level, delta_type total, size_t width) const
            {
                using namespace std;

                out << "[" << setw(level) << "";
                print_line(out, name, length, 100 * length / total, width - level);

                if (children.size()) {
                    delta_type val = length - children_time();
                    double perc = 100.0 * val / total;

                    if (perc > 1e-1) {
                        out << "[" << setw(level + SHIFT_WIDTH) << "";
                        print_line(out, "self", val, perc, width - level - SHIFT_WIDTH);
                    }
                }

                std::map<std::size_t, std::pair<std::string,const profile_unit*>> children_sorted;
                for(typename std::map<std::string, profile_unit>::const_iterator c = children.begin(); c != children.end(); c++)
                {
                    children_sorted[c->second.call_index] = std::make_pair(c->first,&c->second);
                }
                for(auto c = children_sorted.begin(); c != children_sorted.end(); c++)
                {
                    c->second.second->print(out, c->second.first, level + SHIFT_WIDTH, total, width);
                }
            }
            void add_to_totals(std::map<std::string, delta_type> &total_lengths)const
            {
                for(auto c = children.begin(); c != children.end(); c++)
                {
                    if (total_lengths.find(c->first) == total_lengths.end())
                    {
                        total_lengths[c->first] = 0.;
                    }
                    total_lengths[c->first] += c->second.length;
                    c->second.add_to_totals(total_lengths);
                }
            }

            void print_line(std::ostream &out, const std::string &name,
                    delta_type time, double perc, size_t width) const
            {
                using namespace std;

                out << name << ":"
                    << setw(width - name.size()) << ""
                    << setw(10)
                    << fixed << setprecision(3) << time << " " << Event::units()
                    << "] (" << fixed << setprecision(2) << setw(6) << perc << "%)"
                    << endl;
            }

            Event           begin_event;
            delta_type      length;

            /// call_index determines order of children creation inside one profile_unit children.
            /// It is used to safe the call order during prifile output
            std::size_t     call_index;

            std::map<std::string, profile_unit> children;
        };
        // Save ostream flags in constructor, restore in destructor
        struct ios_saver
        {
            std::ios_base &s;
            std::ios_base::fmtflags f;
            std::streamsize p;

            ios_saver(std::ios_base &s)
                : s(s), f(s.flags()), p(s.precision())
            {}

            ~ios_saver()
            {
                s.flags(f);
                s.precision(p);
            }
        };

        std::string name;
        profile_unit root;
        std::vector<profile_unit*> stack;

        void init()
        {
            stack.reserve(128);
            stack.push_back(&root);
            //root.begin = counter.current();
            root.begin_event.record();
        }

        void print(std::ostream &out)
        {
            out << "Profile breakdown:" << std::endl;
            if (stack.back() != &root)
                out << "Warning! Profile is incomplete." << std::endl;
            ios_saver ss(out);
            root.print(out, name, 0, root.length, root.total_width(name, 0));
            out << std::endl;
        }
        void print_totals(std::ostream &out)
        {
            out << "Profile summarize:" << std::endl;
            if (stack.back() != &root)
                out << "Warning! Profile is incomplete." << std::endl;
            std::map<std::string, delta_type> total_lengths;
            root.add_to_totals(total_lengths);
            auto total = root.length;
            auto width = root.total_width(name, 0);
            for(auto c = total_lengths.begin(); c != total_lengths.end(); c++)
            {
                out << "[";
                auto name = c->first;
                auto length = c->second;
                /// NOTE use root print_line here as print_line is infact static (TODO)
                root.print_line(out, name, length, 100 * length / total, width);
            }
            out << std::endl;
        }

        /// Sends formatted profiling data to an output stream.
        /**
         * \param out  Output stream.
         * \param prof Profiler.
         */
        friend std::ostream& operator<<(std::ostream &out, profiler &prof)
        {
            //out << std::endl;
            prof.print(out);
            return out;
        }
};

} // namespace utils

} // namespace scfd

#endif
