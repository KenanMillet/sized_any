// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <kmillet/sized_any/sized_any.hpp>

#include <benchmark/benchmark.h>

template <class AnyT>
static void BM_Any_Empty(benchmark::State& state)
{
    for (auto _ : state) {
        AnyT a;
        benchmark::DoNotOptimize(a);
    }
}

template <class T, class AnyT>
static void BM_Any_Value(benchmark::State& state)
{
    for (auto _ : state) {
        AnyT a(std::in_place_type<T>);
        benchmark::DoNotOptimize(a);
    }
}

template <class T, class AnyT>
static void BM_Any_Copy(benchmark::State& state)
{
    AnyT a = T{};
    for (auto _ : state) {
        AnyT b(a);
        benchmark::DoNotOptimize(b);
    }
}

template <class T, class AnyT>
static void BM_Any_Swap(benchmark::State& state)
{
    AnyT a = T{};
    AnyT b = T{};
    for (auto _ : state) {
        a.swap(b);
    }
}

// Pretty sure I can make this just test move without swap by pre-constructing a ton of objects, but I'm not sure how to do that yet.
// I think it has something to do with state.range and RangeMultiplier/Range like in beman/any_view/all.benchmark.cpp, but I'm not sure how to use it.
template <class T, class AnyT>
static void BM_Any_MoveAndSwap(benchmark::State& state)
{
    AnyT a = T{};
    for (auto _ : state) {
        AnyT b(std::move(a));
        a.reset();
        a.swap(b);
        benchmark::DoNotOptimize(b);
    }
}

template <class AnyT>
static void BM_Any_Cast(benchmark::State& state)
{
    using std::any_cast;
    AnyT a = 42;
    for (auto _ : state) {
        int& ref = any_cast<int&>(a);
        benchmark::DoNotOptimize(ref);
    }
}

template <size_t N>
struct Bytes
{
    Bytes() : data{} { data.fill(0); }
    std::array<char, N> data;
};

#define COMPARE_BENCHMARKS(BenchMark, ...) \
BENCHMARK_TEMPLATE(BenchMark, ##__VA_ARGS__, std::any)->Unit(benchmark::kNanosecond); \
BENCHMARK_TEMPLATE(BenchMark, ##__VA_ARGS__, kmillet::any)->Unit(benchmark::kNanosecond); \
BENCHMARK_TEMPLATE(BenchMark, ##__VA_ARGS__, kmillet::sized_any<32>)->Unit(benchmark::kNanosecond); \
BENCHMARK_TEMPLATE(BenchMark, ##__VA_ARGS__, kmillet::sized_any<64>)->Unit(benchmark::kNanosecond)


COMPARE_BENCHMARKS(BM_Any_Empty);
COMPARE_BENCHMARKS(BM_Any_Value, int);
COMPARE_BENCHMARKS(BM_Any_Value, std::string);
COMPARE_BENCHMARKS(BM_Any_Value, Bytes<32>);
COMPARE_BENCHMARKS(BM_Any_Value, Bytes<64>);
COMPARE_BENCHMARKS(BM_Any_Copy, int);
COMPARE_BENCHMARKS(BM_Any_Copy, std::string);
COMPARE_BENCHMARKS(BM_Any_Copy, Bytes<32>);
COMPARE_BENCHMARKS(BM_Any_Copy, Bytes<64>);
COMPARE_BENCHMARKS(BM_Any_Swap, int);
COMPARE_BENCHMARKS(BM_Any_Swap, std::string);
COMPARE_BENCHMARKS(BM_Any_Swap, Bytes<32>);
COMPARE_BENCHMARKS(BM_Any_Swap, Bytes<64>);
COMPARE_BENCHMARKS(BM_Any_MoveAndSwap, int);
COMPARE_BENCHMARKS(BM_Any_MoveAndSwap, std::string);
COMPARE_BENCHMARKS(BM_Any_MoveAndSwap, Bytes<32>);
COMPARE_BENCHMARKS(BM_Any_MoveAndSwap, Bytes<64>);
COMPARE_BENCHMARKS(BM_Any_Cast);

BENCHMARK_MAIN();