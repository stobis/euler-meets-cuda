#ifndef TIMER_HPP
#define TIMER_HPP

#include <chrono>
#include <string>

class Timer {
private:
  std::clock_t c_start, c_end;
  std::string slug;
  long double overall;

public:
  Timer(std::string);

  void start();
  void stop();
  long double get_ms();
  void print_info(std::string);
  void print_and_restart(std::string);
  void print_overall();
};

#endif // TIMER_HPP
