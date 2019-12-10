#ifndef TEST_RESULT_HPP
#define TEST_RESULT_HPP

#include <vector>

class TestResult {
   private:
    std::vector<short> isBridge;

   public:
    TestResult(int);
    TestResult(std::vector<short>);

    short* data();
    std::vector<short> get_isBridge();

    short& operator[](int);

    void write_to_file(const char*);
    void write_to_stdout();
};

#endif  // TEST_RESULT_HPP
