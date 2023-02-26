#include <gtest/gtest.h>
#include "BitonicSorter.hpp"
#include <random>
#include <algorithm>

const int SMALL_SIZE = 10;
const int BIG_SIZE = 1 << 22;

//------------------------------------------------------------------------------------------------------------------------------


template <typename T>
bool operator == (std::vector<T>& lhs, std::vector<T>& rhs) {
    if (lhs.size() != rhs.size()) return false;

    for (int i = 0; i != lhs.size(); ++i) {
        if (lhs[i] != rhs[i]) 
            return false;
    }
    return true;
}

//------------------------------------------------------------------------------------------------------------------------------

template <typename T> 
void TestBody(size_t size, OpenCLApp::SortDirection direction) {
    OpenCLApp::BitonicSorter<T> sort;

    auto rigth_border = std::numeric_limits<T>::max();
    auto left_border  = std::numeric_limits<T>::lowest();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<T> random(left_border, rigth_border);
    
    std::vector<T> data(size);
    for (auto& x: data) x = random(gen);

    std::vector<T> copy = data;
    sort(data.begin(), data.end(), direction);

    if (direction == OpenCLApp::INCREASING)
        std::sort(copy.begin(), copy.end());
    else 
        std::sort(copy.begin(), copy.end(), std::greater());

    EXPECT_EQ(data, copy);
}


//------------------------------------------------------------------------------------------------------------------------------



template <> 
void TestBody<float>(size_t size, OpenCLApp::SortDirection direction) {
    using T = float;
    
    OpenCLApp::BitonicSorter<T> sort;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> random{};
    
    std::vector<T> data(size);
    for (auto& x: data) x = random(gen);

    std::vector<T> copy = data;
    sort(data.begin(), data.end(), direction);

    if (direction == OpenCLApp::INCREASING)
        std::sort(copy.begin(), copy.end());
    else 
        std::sort(copy.begin(), copy.end(), std::greater());

    EXPECT_EQ(data, copy);
}

//------------------------------------------------------------------------------------------------------------------------------


template <> 
void TestBody<double>(size_t size, OpenCLApp::SortDirection direction) {
    using T = double;
    
    OpenCLApp::BitonicSorter<T> sort;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> random{};
    
    std::vector<T> data(size);
    for (auto& x: data) x = random(gen);

    std::vector<T> copy = data;
    sort(data.begin(), data.end(), direction);

    if (direction == OpenCLApp::INCREASING)
        std::sort(copy.begin(), copy.end());
    else 
        std::sort(copy.begin(), copy.end(), std::greater());

    EXPECT_EQ(data, copy);
}


//------------------------------------------------------------------------------------------------------------------------------


#define TEST_CREATER(type)                                          \
    TEST(BitonicSortTest, test_##type##_1) {                        \
        ::TestBody<type>(SMALL_SIZE, OpenCLApp::INCREASING);        \
    }                                                               \
                                                                    \
    TEST(BitonicSortTest, test_##type##_2) {                        \
        ::TestBody<type>(BIG_SIZE, OpenCLApp::INCREASING);          \
    }                                                               \
                                                                    \
    TEST(BitonicSortTest, test_##type##_3) {                        \
        ::TestBody<type>(SMALL_SIZE, OpenCLApp::DECREASING);        \
    }                                                               \
                                                                    \
    TEST(BitonicSortTest, test_##type##_4) {                        \
        ::TestBody<type>(BIG_SIZE, OpenCLApp::DECREASING);          \
    }                                                               


//------------------------------------------------------------------------------------------------------------------------------

TEST_CREATER(float)

TEST_CREATER(double)

TEST_CREATER(int)

TEST_CREATER(unsigned)

TEST_CREATER(int64_t)

TEST_CREATER(uint64_t)


//------------------------------------------------------------------------------------------------------------------------------

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

