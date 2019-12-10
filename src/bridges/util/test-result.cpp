#include "test-result.hpp"
#include <fstream>
#include <iostream>

TestResult::TestResult(int size) : isBridge(std::vector<short>(size, 0)) {}

TestResult::TestResult(std::vector<short> data) : isBridge(data) {}

short* TestResult::data() { return isBridge.data(); }

std::vector<short> TestResult::get_isBridge() { return isBridge; }

void TestResult::write_to_file(const char* filename) {
    std::ofstream out(filename, std::ios::binary);
    out.write(reinterpret_cast<char*>(isBridge.data()), sizeof(short));
}

void TestResult::write_to_stdout() {
    for (auto& res : isBridge) {
        std::cout << res << std::endl;
    }
}

short& TestResult::operator[](int x) { return isBridge[x]; }
