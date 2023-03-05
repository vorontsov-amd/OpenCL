#include "BitonicSorter.hpp"
#include <random>
#include <iostream>
#include <algorithm>
#include <chrono>

#ifndef TYPE
    #define TYPE int
#endif


#ifndef PLATFORM
    #define PLATFORM OpenCLApp::Platform::NVIDIA
#endif


int main() try {
    using namespace std::chrono;
    using fseconds = duration<float>;

    OpenCLApp::BitonicSorter<int> sort(PLATFORM);

    auto rigth_border = std::numeric_limits<int>::max();
    auto left_border  = std::numeric_limits<int>::lowest();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> random(left_border, rigth_border);
    
    std::vector<int> data(1 << 22);
    for (auto& x: data) x = random(gen);

    std::vector<int> copy = data;
    auto start = system_clock::now();
    sort(data.begin(), data.end());
    auto end = system_clock::now() - start;
    std::cout << "runtime " << duration_cast<fseconds>(end).count() << "c\n";


    start = system_clock::now();
    std::sort(copy.begin(), copy.end());
    end = system_clock::now() - start;
    std::cout << "runtime " << duration_cast<fseconds>(end).count() << "c\n";

    
    // size_t size = 0;
    // std::cin >> size;

    // std::vector<TYPE> data(size);
    // for (size_t i = 0; i < size; ++i) {
    //     std::cin >> data[i];
    // }

    // OpenCLApp::BitonicSorter<TYPE> sort(PLATFORM);
    // sort(data.begin(), data.end());

    // for (auto& x: data) {
    //     std::cout << x << ' ';
    // }
    // std::cout << std::endl;
}

catch (cl::Error& error) {
    std::cout << error.what() << ". Error code = " << error.err() << '\n';
}
catch (std::exception& error) {
    std::cout << error.what() << "\n";
}


