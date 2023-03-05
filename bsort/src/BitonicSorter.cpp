#include "BitonicSorter.hpp"
#include <random>
#include <iostream>
#include <algorithm>
#include <chrono>

#ifndef TYPE
    #define TYPE float
#endif


#ifndef PLATFORM
    #define PLATFORM NVIDIA
#endif

#define USE_PLATFORM OpenCLApp::Platform::PLATFORM


int main() try {

    using namespace std::chrono;
    using fseconds = duration<float>;


    OpenCLApp::BitonicSorter<TYPE> sort(USE_PLATFORM);

    auto rigth_border = std::numeric_limits<TYPE>::max() / 10;
    auto left_border  = std::numeric_limits<TYPE>::lowest() / 10;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<TYPE> random(left_border, rigth_border);

    std::vector<float> my_time;
    std::vector<float> std_time;

    for (size_t i = 20; i <= 25; ++i) {
        size_t size = 1 << i;
 
        std::vector<TYPE> data(size);
        for (auto& x: data) x = random(gen);

        std::vector<TYPE> copy = data;
        auto start = system_clock::now();
        sort(data.begin(), data.end());
        auto end = system_clock::now() - start;
        my_time.push_back(duration_cast<fseconds>(end).count());
        start = system_clock::now();
        std::sort(copy.begin(), copy.end());
        end = system_clock::now() - start;
        std_time.push_back(duration_cast<fseconds>(end).count());
    }

    std::ofstream out("data.txt");
    assert(out.is_open());
    for (auto x: my_time) {
        out << x;
        out << '\n';
    }
    for (auto x: std_time) {
        out << x;
        out << '\n';
    }

    out.close();
    // size_t size = 0;
    // std::cin >> size;

    // std::vector<TYPE> data(size);
    // for (size_t i = 0; i < size; ++i) {
    //     std::cin >> data[i];
    // }

    // OpenCLApp::BitonicSorter<TYPE> sort(USE_PLATFORM);
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


