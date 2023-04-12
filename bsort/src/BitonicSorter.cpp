#include "BitonicSorter.hpp"
#include <algorithm>
#include <boost/program_options.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/value_semantic.hpp>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <ostream>
#include <random>
#include <string>
#include <vector>

#ifndef TYPE
#define TYPE int
#endif

using T = int;

#ifndef PLATFORM
#define PLATFORM NVIDIA
#endif

#define USE_PLATFORM OpenCLApp::Platform::PLATFORM

namespace po  = boost::program_options;
namespace chr = std::chrono;

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

auto fillData(std::size_t size, bool isTestMode) {
  std::vector<T> data(size);
  if (isTestMode) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> rand(-1000, 1000);
    for (size_t i = 0; i < size; ++i) {
      data[i] = rand(gen);
    }
  } else {
    for (size_t i = 0; i < size; ++i) {
      std::cin >> data[i];
    }
  }
  return data;
}

void RunTestProgram(std::string platformName, std::size_t size) {
  std::vector<T> data(size);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> rand(-1000, 1000);

  for (auto&& x : data) x = rand(gen);
  std::vector dataCopy = data;

  auto start = chr::high_resolution_clock::now();
  std::sort(dataCopy.begin(), dataCopy.end());
  auto end   = chr::high_resolution_clock::now();
  std::cout << "std::sort() time: "
            << std::chrono::duration_cast<chr::duration<float>>(end - start).count()
            << " sec." << std::endl;

  OpenCLApp::BitonicSorter<T> sort(platformName);
  start = chr::high_resolution_clock::now();
  sort(data.begin(), data.end());
  end   = chr::high_resolution_clock::now();
  std::cout << "My bsort() time: "
            << std::chrono::duration_cast<chr::duration<float>>(end - start).count()
            << " sec." << std::endl;
}

int main(int ac, const char **av) try {
  auto [platform, size, isTestMode] = ParseConsoleArgument(ac, av);

  if (isTestMode) {
    RunTestProgram(platform, size);
  }
//  } else {
//    RunGeneralProgram();
//  }
//
//  auto&& data = fillData(size, isTestMode);
//
//  OpenCLApp::BitonicSorter<T> sort(platform);
//  sort(data.begin(), data.end());
//
//  for (auto &x : data) {
//    std::cout << x << ' ';
//  }
//  std::cout << std::endl;
}

catch (cl::Error &error) {
  std::cout << error.what() << ". Error code = " << error.err() << '\n';
} catch (std::exception &error) {
  std::cout << error.what() << "\n";
}

