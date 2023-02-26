#include "BitonicSorter.hpp"
#include <random>
#include <iostream>
#include <algorithm>
#include <chrono>


int main() try {
    using type = float;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> random(-1000, 1000);
    std::vector<type> data(1 << 22);
    for (auto& x: data) {
        x = random(gen);
    }

    std::vector<type> data2 {data};

    // for (auto& x: data) {
    //     std::cout << x << ' ';
    // }

    OpenCLApp::BitonicSorter<type> sort;

    auto begin = std::chrono::system_clock::now();
    sort(data.begin(), data.end());
    auto end = std::chrono::system_clock::now() - begin;
    std::cout << std::chrono::duration_cast<std::chrono::duration<float>> (end).count() << "\n";
    begin = std::chrono::system_clock::now();
    std::sort(data2.begin(), data2.end());

    // for (int i = 0; i < data.size(); ++i) {
    //     if (data[i] != data2[i]) {
    //         std::cout << "FAIL";
    //     }
    // }

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


