#ifndef TIMER_HPP
#define TIMER_HPP

#include <chrono>
#include <string>

class Timer {
private:
  std::chrono::nanoseconds start_ns, end_ns;
  std::string slug;
  long double overall;

public:
  Timer(std::string);

  void start();
  void stop();
  long double get_ms();
  long double stop_and_get_ms();
  void print_info(std::string);
  void print_and_restart(std::string);
  void print_overall();
};

#endif // TIMER_HPP
