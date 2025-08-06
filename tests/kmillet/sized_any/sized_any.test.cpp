// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <kmillet/sized_any/sized_any.hpp>

#include <gtest/gtest.h>

#include <string>
#include <vector>

using kmillet::sized_any;
using kmillet::any;
using kmillet::any_cast;

TEST(SizedAnyTest, Empty)
{
    sized_any<32> a;
    EXPECT_FALSE(a.has_value());
    EXPECT_EQ(a.type(), typeid(void));
}

TEST(SizedAnyTest, Int)
{
    sized_any<32> a = 42;
    EXPECT_TRUE(a.has_value());
    EXPECT_EQ(a.type(), typeid(int));
    EXPECT_EQ(any_cast<int>(a), 42);
    EXPECT_EQ(a.capacity(), 32);
}

TEST(SizedAnyTest, String)
{
    auto a = kmillet::make_sized_any<64, std::string>("hello world!");
    EXPECT_TRUE(a.has_value());
    EXPECT_EQ(a.type(), typeid(std::string));
    EXPECT_EQ(any_cast<const std::string&>(a), "hello world!");
    EXPECT_EQ(a.capacity(), 64);
}

TEST(SizedAnyTest, Move)
{
    sized_any<32> a = 123;
    sized_any<32> b = std::move(a);
    EXPECT_TRUE(b.has_value());
    EXPECT_EQ(any_cast<int>(b), 123);
    EXPECT_FALSE(a.has_value());
}

TEST(SizedAnyTest, Copy)
{
    sized_any<32> a = 55;
    sized_any<32> b = a;
    EXPECT_TRUE(b.has_value());
    EXPECT_EQ(any_cast<int>(b), 55);
    EXPECT_TRUE(a.has_value());
}

TEST(SizedAnyTest, Reset) 
{
    sized_any<32> a = 99;
    a.reset();
    EXPECT_FALSE(a.has_value());
}

TEST(SizedAnyTest, Swap)
{
    sized_any<32> a = 1;
    sized_any<32> b = 2;
    a.swap(b);
    EXPECT_EQ(any_cast<int>(a), 2);
    EXPECT_EQ(any_cast<int>(b), 1);
}

TEST(SizedAnyTest, Emplace)
{
    sized_any<64> a;
    a.emplace<std::vector<int>>({1,2,3});
    auto& v = any_cast<std::vector<int>&>(a);
    EXPECT_EQ(v.size(), 3);
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 2);
    EXPECT_EQ(v[2], 3);
}

TEST(SizedAnyTest, BadCast)
{
    sized_any<32> a = 42;
    EXPECT_THROW(any_cast<std::string>(a), std::bad_any_cast);
}

TEST(SizedAnyTest, AnyAlias)
{
    EXPECT_TRUE(sizeof(any) == sizeof(std::any));
    auto a = kmillet::make_any<std::string>("test");
    EXPECT_TRUE(a.has_value());
    EXPECT_EQ(a.type(), typeid(std::string));
    EXPECT_EQ(any_cast<std::string>(a), "test");
    EXPECT_TRUE(typeid(a) == typeid(kmillet::any));
}

TEST(SizedAnyTest, MakeSizedAnyToFit)
{
    // Test with a type that requires less space than a pointer
    auto a = kmillet::make_sized_any<char>('a');
    EXPECT_EQ(a.type(), typeid(char));
    EXPECT_EQ(a.capacity(), sizeof(void*));

    // Test with a type that requires more space than a pointer
    struct MyStruct { void* x; void* y; };
    auto b = kmillet::make_sized_any<MyStruct>();
    EXPECT_EQ(b.type(), typeid(MyStruct));
    EXPECT_EQ(b.capacity(), sizeof(MyStruct));
}