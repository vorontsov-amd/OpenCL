#include "BitonicSorter.hpp"
#include <random>
#include <iostream>
#include <algorithm>
#include <chrono>

#ifndef TYPE
    #define TYPE int
#endif

int main() try {
    size_t size = 0;
    std::cin >> size;

    std::vector<TYPE> data(size);
    for (size_t i = 0; i < size; ++i) {
        std::cin >> data[i];
    }

    OpenCLApp::BitonicSorter<TYPE> sort;
    sort(data.begin(), data.end());

    for (auto& x: data) {
        std::cout << x << ' ';
    }
    std::cout << std::endl;
}

catch (cl::Error& error) {
    std::cout << error.what() << ". Error code = " << error.err() << '\n';
}
catch (std::exception& error) {
    std::cout << error.what() << "\n";
}


