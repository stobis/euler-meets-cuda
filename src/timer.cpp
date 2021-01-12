#include "timer.hpp"
#include <iostream>


std::chrono::nanoseconds getCurrentNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch());
}

Timer::Timer(std::string slug) : slug(slug), start_ns(getCurrentNs()), overall(0) {}

void Timer::start() { start_ns = getCurrentNs(); }

void Timer::stop() { end_ns = getCurrentNs(); }


long double Timer::get_ms() { 
  return (end_ns.count() - start_ns.count()) / 1e6;
  }

long double Timer::stop_and_get_ms() {
  stop();
  return get_ms();
}

void Timer::print_info(std::string desc) {
  std::cout.precision(3);
  std::cout << std::fixed << slug << ": " << desc << ": " << get_ms() << " ms." << std::endl;
}

void Timer::print_and_restart(std::string desc) {
  stop();
  overall += get_ms();
  print_info(desc);
  start();
}

void Timer::print_overall() {
  std::cout.precision(3);
  std::cout << std::fixed << slug << ": "
            << "Overall"
            << ": " << overall << " ms." << std::endl;
}
