#include "BitonicSorter.hpp"
#include <random>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <boost/program_options.hpp>

#ifndef TYPE
    #define TYPE int
#endif


#ifndef PLATFORM
    #define PLATFORM NVIDIA
#endif

#define USE_PLATFORM OpenCLApp::Platform::PLATFORM

namespace po = boost::program_options;

auto ParseConsoleArgument(int ac, const char** av) {
  std::string platform;
  std::size_t size = 0;
  bool isTestMode = true;

  po::options_description desc("Allowed options");
  desc.add_options()
      ("help", "produce help message")
      ("platform", po::value<std::string>(&platform)->default_value("NVIDIA"), "platform for computing. By defaul NVIDIA")
      ("test", po::value<std::size_t>(&size), "the amount of elements to generate N random numbers")
  ;

  po::variables_map vm;
  po::store(po::parse_command_line(ac, av, desc), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << desc << "\n";
    std::exit(0);
  }
  if (vm.count("test")) {
    std::cout << "Number of random elements for test " << size << ".\n";
  } else {
    isTestMode = false;
    std::cin >> size;
  }
  return std::make_tuple(platform, size, isTestMode);
}


int main(int ac, const char** av) try {

    auto [platform, size, isTestMode] = ParseConsoleArgument(ac, av);

    using namespace std::chrono;
    using fseconds = duration<float>;


    OpenCLApp::BitonicSorter<TYPE> sort(USE_PLATFORM);

    auto rigth_border = std::numeric_limits<TYPE>::max() / 10;
    auto left_border  = std::numeric_limits<TYPE>::lowest() / 10;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<TYPE> random(left_border, rigth_border);

    std::vector<TYPE> data(size);
    for (auto& x: data) x = random(gen);

    std::vector<TYPE> copy = data;
    auto start = system_clock::now();
    sort(data.begin(), data.end());
    auto end = system_clock::now() - start;
    auto my_time = std::chrono::duration_cast<fseconds>(end).count();
    start = system_clock::now();
    std::sort(copy.begin(), copy.end());
    end = system_clock::now() - start;
    auto std_time = std::chrono::duration_cast<fseconds>(end).count();

    std::cout  << "my time: \n" << my_time << '\n';
    std::cout  << "std time: \n" << std_time << '\n';
}

catch (cl::Error& error) {
    std::cout << error.what() << ". Error code = " << error.err() << '\n';
}
catch (std::exception& error) {
    std::cout << error.what() << "\n";
}


