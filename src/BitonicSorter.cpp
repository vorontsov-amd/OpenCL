#include "BitonicSorter.hpp"
#include <random>
#include <iostream>
#include <algorithm>
#include <chrono>


int main() try {

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> random(0, 10000000);
    std::vector<float> data(1 << 24);
    for (auto& x: data) {
        x = random(gen);
    }

    std::vector<float> data2 {data};

    // for (auto& x: data) {
    //     std::cout << x << ' ';
    // }

    OpenCLApp::BitonicSorter<float> sort;

    auto begin = std::chrono::system_clock::now();
    sort(data.begin(), data.end());
    auto end = std::chrono::system_clock::now() - begin;
    std::cout << std::chrono::duration_cast<std::chrono::duration<float>> (end).count() << "\n";
    begin = std::chrono::system_clock::now();
    std::sort(data.begin(), data.end());
    end = std::chrono::system_clock::now() - begin;
    std::cout << std::chrono::duration_cast<std::chrono::duration<float>> (end).count() << "\n";
    // for (auto& x: data) {
    //     std::cout << x << ' ';
    // }


}

catch (cl::Error& error) {
    std::cout << error.what() << ". Error code = " << error.err() << '\n';
}
catch (std::exception& error) {
    std::cout << error.what() << "\n";
}


