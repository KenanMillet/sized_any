/**
 * @file sized_any.hpp
 * @author Kenan Millet
 * @brief A stack-optimized, type-erased container similar to `std::any`, with customizable buffer size and efficient memory usage.
 *
 * This header provides the `kmillet::sized_any<N>` class template, a drop-in alternative to `std::any` that allows control over the internal buffer size.
 * - For types up to `N` bytes and that are noexcept-movable, storage is in-place (no heap allocation).
 * - For larger or non-noexcept-movable types, heap allocation is used.
 * - Strong exception safety and type-safe access via `kmillet::any_cast<T>`.
 * - Helper functions `kmillet::make_sized_any` and `kmillet::make_any` mirror the standard library's `std::make_any`.
 * - The alias `kmillet::any` provides a direct replacement for `std::any` with the same buffer size.
 *
 * @section Usage
 * @code
 * kmillet::sized_any<16> a = 42; // stores an int in-place without heap allocation since sizeof(int) <= 16
 * int value = kmillet::any_cast<int>(a); // access the int value or throws std::bad_any_cast if type does not match
 * std::string& a_ref = a.emplace<std::string>("hello"); // will likely dynamic-allocate since sizeof(std::string) > 16 on most platforms
 * auto b = kmillet::make_sized_any<64, std::vector<int>>({1, 2, 3}); // b is a kmillet::sized_any<64> holding a std::vector<int> with 3 elements
 * auto* b_ptr = kmillet::any_cast<std::vector<int>>(&b); // safely access the vector as a pointer
 * auto c = kmillet::make_sized_any<std::string>("world"); // c is a kmillet::sized_any<sizeof(std::string)>
 * auto& c_ref = kmillet::any_cast<std::string&>(c); // access the string by reference or throws std::bad_any_cast if type does not match
 * c.reset(); // destroys the contained string
 * kmillet::any d; // kmillet::any has the same in-place storage capacity as std::any
 * d = kmillet::make_any<std::string>("example"); // will dynamic-allocate if std::make_any<std::string> would
 * @endcode
 *
 * @section Deviations
 * - The `kmillet::sized_any<N>::emplace` method may reuse heap-allocated storage for performance, which is a deliberate deviation from the strict behavior of `std::any::emplace`.
 * - Unlike `std::any`, which leaves the source object in a valid but unspecified state after a move construct/assignment, `kmillet::sized_any<N>` leaves the source in an empty state equivalent to a default-constructed `kmillet::sized_any<N>`.
 *
 * @section License
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 * See the LICENSE file in the root of this repository for complete details.
 */

#pragma once

#include <any>
#include <array>
#include <concepts>
#include <cstddef>

namespace kmillet
{
    template <std::size_t N> class sized_any;

    namespace sized_any_detail
    {
        template<class T> struct is_sized_any : std::false_type {};
        template<std::size_t N> struct is_sized_any<sized_any<N>> : std::true_type {};

        template<class T> struct is_in_place_type : std::false_type {};
        template<class T> struct is_in_place_type<std::in_place_type_t<T>> : std::true_type {};

        struct ITypeInfo;
        template<class T> struct TypeInfo;
        template<class T> inline constexpr TypeInfo<T> info{};
    }

    /**
     * @brief The concept `kmillet::sized_any_optimized<T, N>` is satisfied if and only if a
     *        `kmillet::sized_any<N>` will not perform dynamic allocation to hold an object of type `std::decay_t<T>`.
     *
     * This means that `sizeof(std::decay_t<T>) <= N` and `std::is_nothrow_move_constructible_v<std::decay_t<T>>` is true.
     * @tparam T The type to check.
     * @tparam N The size of the buffer.
     */
    template<class T, std::size_t N>
    concept sized_any_optimized = !sized_any_detail::info<std::decay_t<T>>.needsAlloc(N);

    /**
     * @brief A type-erased container similar to `std::any`, but with a template-sized buffer for in-place allocation.
     *
     * Intended as a drop-in, more efficient alternative to `std::any` with better control over memory usage.
     * - No heap allocation for types up to `N` bytes in size that are noexcept-movable; otherwise, heap allocation is used.
     *   - `N` must be at least the size of a pointer to ensure that it can hold a pointer to heap-allocated memory if needed.
     * - Provides strong exception safety and type-safe access via `kmillet::any_cast<T>`.
     * - `kmillet::make_sized_any` functions analogous to `std::make_any` are provided for convenient construction.
     * - `kmillet::any` alias is provided as a direct replacement of `std::any` that is compatible with other specializations of `kmillet::sized_any`.
     * - `kmillet::make_any` functions are provided as a direct replacement of `std::make_any`.
     *
     * See documentation below for usage and design details.
     * @tparam N The size of the buffer used for in-place storage.
     */
    template<std::size_t N>
    class sized_any
    {
        // The size of the buffer used must be at least the size of a pointer to ensure that it can hold a pointer to heap-allocated memory if needed.
        static_assert(N >= sizeof(void*));
    public:
        /**
         * @brief Constructs an empty object.
         */
        constexpr sized_any() noexcept;
        /**
         * @brief Copies the content of `other` into a new instance.
         * @param other The `kmillet::sized_any<N>` to copy.
         * @details No dynamic allocation will occur if the content of `other` satisfies `kmillet::sized_any_optimized<N>`.
         */
        sized_any(const sized_any& other);
        /**
         * @brief Copies the content of `other` into a new instance.
         * @tparam M The size of the buffer used by `other`.
         * @param other The `kmillet::sized_any<M>` to copy.
         * @details No dynamic allocation will occur if the content of `other` satisfies `kmillet::sized_any_optimized<N>`.
         */
        template<std::size_t M>
        sized_any(const sized_any<M>& other);
        /**
         * @brief Moves the content of `other` into a new instance.
         * @param other The `kmillet::sized_any<N>` to move.
         * @details Unlike `std::any`, which leaves `other` in a valid but unspecified state after the move,
         * this implementation leaves `other` in a state that is equivalent to an empty `kmillet::sized_any<N>`.
         * No dynamic allocation will occur.
         * If the content of `other` satisfies `kmillet::sized_any_optimized<N>`, then it will be moved into the buffer of this instance.
         * Otherwise, the pointer held by `other` that is pointing to the heap will be moved into this instance.
         */
        sized_any(sized_any&& other) noexcept;
        /**
         * @brief Moves the content of `other` into a new instance.
         * @tparam M The size of the buffer used by `other`.
         * @param other The `kmillet::sized_any<M>` to move.
         * @details Unlike `std::any`, which leaves `other` in a valid but unspecified state after the move,
         * this implementation leaves `other` in a state that is equivalent to an empty `kmillet::sized_any<M>`.
         * Dynamic allocation will occur if and only if the content of `other` does not satisfy `kmillet::sized_any_optimized<N>`, but it does satisfy `kmillet::sized_any_optimized<M>`.
         * This means that if `M < N`, then no dynamic allocation will occur.
         */
        template<std::size_t M>
        sized_any(sized_any<M>&& other) noexcept(M < N);
        /**
         * @brief Constructs an object with initial content of type `std::decay_t<ValueType>`, direct-initialized from `std::forward<ValueType>(value)`.
         * @tparam ValueType The type of the value to be stored.
         * @param value The value to be stored.
         * @details Requires that `std::decay_t<ValueType>` is not a specialization of `kmillet::sized_any` nor a specialization of `std::in_place_type_t`, and is copy-constructible.
         * Noexcept so long as constructing `std::decay_t<ValueType>` from `std::forward<ValueType>(value)` is noexcept and `kmillet::sized_any_optimized<ValueType, N>` is satisfied.
         */
        template<class ValueType>
        requires(std::conjunction_v<std::negation<sized_any_detail::is_sized_any<std::decay_t<ValueType>>>, std::negation<sized_any_detail::is_in_place_type<std::decay_t<ValueType>>>, std::is_copy_constructible<std::decay_t<ValueType>>>)
        sized_any(ValueType&& value) noexcept(std::is_nothrow_constructible_v<std::decay_t<ValueType>, ValueType> && sized_any_optimized<ValueType, N>);
        /**
         * @brief Constructs an object with initial content of type `std::decay_t<ValueType>`, direct-non-list-initialized from `std::forward<Args>(args)...`.
         * @tparam ValueType The type of the value to be stored.
         * @tparam Args The types of the arguments to be forwarded to the constructor of `std::decay_t<ValueType>`.
         * @param args The arguments to be forwarded to the constructor of `std::decay_t<ValueType>`.
         * @details Requires that `std::decay_t<ValueType>` is copy-constructible and constructible from `std::forward<Args>(args)...`.
         * Noexcept so long as constructing `std::decay_t<ValueType>` direct-non-list-initialized from `std::forward<Args>(args)...` is noexcept and `kmillet::sized_any_optimized<ValueType, N>` is satisfied.
         */
        template<class ValueType, class... Args>
        requires(std::copy_constructible<std::decay_t<ValueType>> && std::constructible_from<std::decay_t<ValueType>, Args...>)
        explicit sized_any(std::in_place_type_t<ValueType>, Args&&... args) noexcept(std::is_nothrow_constructible_v<std::decay_t<ValueType>, Args...> && sized_any_optimized<ValueType, N>);
        /**
         * @brief Constructs an object with initial content of type `std::decay_t<ValueType>`, direct-non-list-initialized from `il, std::forward<Args>(args)...`.
         * @tparam ValueType The type of the value to be stored.
         * @tparam U The type of the elements in the initializer list.
         * @tparam Args The types of the arguments to be forwarded to the constructor of `std::decay_t<ValueType>`.
         * @param il The initializer list to be used for constructing the object.
         * @param args The arguments to be forwarded to the constructor of `std::decay_t<ValueType>`.
         * @details Requires that `std::decay_t<ValueType>` is copy-constructible and constructible from `il, std::forward<Args>(args)...`.
         * Noexcept so long as constructing `std::decay_t<ValueType>` direct-non-list-initialized from `il, std::forward<Args>(args)...` is noexcept and `kmillet::sized_any_optimized<ValueType, N>` is satisfied.
         */
        template<class ValueType, class U, class... Args>
        requires(std::copy_constructible<std::decay_t<ValueType>> && std::constructible_from<std::decay_t<ValueType>, std::initializer_list<U>&, Args...>)
        explicit sized_any(std::in_place_type_t<ValueType>, std::initializer_list<U> il, Args&&... args) noexcept(std::is_nothrow_constructible_v<std::decay_t<ValueType>, std::initializer_list<U>&, Args...> && sized_any_optimized<ValueType, N>);

        /**
         * @brief Destroys the contained object, if any, as if by a call to `reset()`.
         */
        ~sized_any();

        /**
         * @brief Assigns by copying the state of `rhs`, as if by `kmillet::sized_any<N>(rhs).swap(*this)`.
         * @param rhs The `kmillet::sized_any<N>` to copy.
         * @return A reference to `*this`.
         */
        sized_any& operator=(const sized_any& rhs);
        /**
         * @brief Assigns by copying the state of `rhs`, as if by `kmillet::sized_any<N>(rhs).swap(*this)`.
         * @tparam M The size of the buffer used by `rhs`.
         * @param rhs The `kmillet::sized_any<M>` to copy.
         * @return A reference to `*this`.
         */
        template<std::size_t M>
        sized_any& operator=(const sized_any<M>& rhs);
        /**
         * @brief Assigns by moving the state of `rhs`, as if by `kmillet::sized_any<N>(std::move(rhs)).swap(*this)`.
         * @param rhs The `kmillet::sized_any<N>` to move.
         * @return A reference to `*this`.
         * @details Unlike `std::any`, which leaves `other` in a valid but unspecified state after the assignment,
         * this implementation leaves `other` in a state that is equivalent to an empty `kmillet::sized_any<N>`.
         */
        sized_any& operator=(sized_any&& rhs) noexcept;
        /**
         * @brief Assigns by moving the state of `rhs`, as if by `kmillet::sized_any<N>(std::move(rhs)).swap(*this)`.
         * @tparam M The size of the buffer used by `rhs`.
         * @param rhs The `kmillet::sized_any<M>` to move.
         * @return A reference to `*this`.
         * @details Unlike `std::any`, which leaves `other` in a valid but unspecified state after the assignment,
         * this implementation leaves `other` in a state that is equivalent to an empty `kmillet::sized_any<N>`.
         * Dynamic allocation will occur if and only if the content of `other` does not satisfy `kmillet::sized_any_optimized<N>`, but it does satisfy `kmillet::sized_any_optimized<M>`.
         * This means that if `M < N`, then no dynamic allocation will occur.
         */
        template<std::size_t M>
        sized_any& operator=(sized_any<M>&& rhs) noexcept(M < N);
        /**
         * @brief Assigns the type and value of `rhs`, as if by `kmillet::sized_any<N>(std::forward<ValueType>(rhs)).swap(*this)`.
         * @tparam ValueType The type of the value to be assigned.
         * @param rhs The value to be assigned.
         * @return A reference to `*this`.
         * @details Requires that `std::decay_t<ValueType>` is not a specialization of `kmillet::sized_any` nor a specialization of `std::in_place_type_t`, and is copy-constructible.
         */
        template<class ValueType>
        requires(std::conjunction_v<std::negation<sized_any_detail::is_sized_any<std::decay_t<ValueType>>>, std::is_copy_constructible<std::decay_t<ValueType>>>)
        sized_any& operator=(ValueType&& rhs) noexcept(noexcept(sized_any{std::forward<ValueType>(rhs)}));

        /**
         * @brief Changes the contained object to one of type `std::decay_t<ValueType>` constructed from the arguments.
         * @tparam ValueType The type of the value to be stored.
         * @tparam Args The types of the arguments to be forwarded to the constructor of `std::decay_t<ValueType>`.
         * @param args The arguments to be forwarded to the constructor of `std::decay_t<ValueType>`.
         * @return A reference to the newly constructed object of type `std::decay_t<ValueType>`.
         * @details First, destroys the contained object, if any. Unlike `std::any`, which always deallocates heap storage on `emplace` by calling `reset()`,
         * this implementation may reuse existing heap-allocated storage if both the current and new contained types require allocation
         * and have the same size. This optimization improves performance but is a deliberate deviation from `std::any`'s strict behavior.
         * Then, constructs an object with initial content of type `std::decay_t<ValueType>`, direct-non-list-initialized from `std::forward<Args>(args)...`.
         * Requires that `std::decay_t<ValueType>` is copy-constructible and is constructible from `std::forward<Args>(args)...`.
         * Noexcept so long as constructing `std::decay_t<ValueType>` direct-non-list-initialized from `std::forward<Args>(args)...` is noexcept and `kmillet::sized_any_optimized<ValueType, N>` is satisfied.
         */
        template<class ValueType, class... Args>
        requires(std::copy_constructible<std::decay_t<ValueType>> && std::constructible_from<std::decay_t<ValueType>, Args...>)
        std::decay_t<ValueType>& emplace(Args&&... args) noexcept(noexcept(sized_any{std::in_place_type<ValueType>, std::forward<Args>(args)...}));
        /**
         * @brief Changes the contained object to one of type `std::decay_t<ValueType>` constructed from the arguments.
         * @tparam ValueType The type of the value to be stored.
         * @tparam U The type of the elements in the initializer list.
         * @tparam Args The types of the arguments to be forwarded to the constructor of `std::decay_t<ValueType>`.
         * @param il The initializer list to be used for constructing the object.
         * @param args The arguments to be forwarded to the constructor of `std::decay_t<ValueType>`.
         * @return A reference to the newly constructed object of type `std::decay_t<ValueType>`.
         * @details First, destroys the contained object, if any. Unlike `std::any`, which always deallocates heap storage on `emplace` by calling `reset()`,
         * this implementation may reuse existing heap-allocated storage if both the current and new contained types require allocation
         * and have the same size. This optimization improves performance but is a deliberate deviation from `std::any`'s strict behavior.
         * Then, constructs an object with initial content of type `std::decay_t<ValueType>`, direct-non-list-initialized from `il, std::forward<Args>(args)...`.
         * Requires that `std::decay_t<ValueType>` is copy-constructible and is constructible from `il, std::forward<Args>(args)...`.
         * Noexcept so long as constructing `std::decay_t<ValueType>` direct-non-list-initialized from `il, std::forward<Args>(args)...` is noexcept and `kmillet::sized_any_optimized<ValueType, N>` is satisfied.
         */
        template<class ValueType, class U, class... Args>
        requires(std::copy_constructible<std::decay_t<ValueType>> && std::constructible_from<std::decay_t<ValueType>, std::initializer_list<U>&, Args...>)
        std::decay_t<ValueType>& emplace(std::initializer_list<U> il, Args&&... args) noexcept(noexcept(sized_any{std::in_place_type<ValueType>, il, std::forward<Args>(args)...}));

        /**
         * @brief If `*this` contains a value, destroys the contained value.
         * @details `*this` does not contain a value after this call.
         */
        void reset() noexcept;
        /**
         * @brief Swaps the content of two `kmillet::sized_any<N>` objects.
         * @param other The `kmillet::sized_any<N>` to swap with.
         */
        void swap(sized_any& other) noexcept;
        /**
         * @brief Swaps the content of a `kmillet::sized_any<N>` object with a `kmillet::sized_any<M>` object.
         * @tparam M The size of the buffer used by `other`.
         * @param other The `kmillet::sized_any<M>` to swap with.
         */
        template<std::size_t M>
        void swap(sized_any<M>& other);

        /**
         * @brief Gets the in-place storage capacity of the `kmillet::sized_any<N>` instance.
         * @return The size of the buffer (in bytes) used to hold the contained value, which is `N`.
         */
        [[nodiscard]] static constexpr std::size_t capacity() noexcept { return N; }
        /**
         * @brief Checks whether the object contains a value.
         * @returns `true` if and only if the instance is non-empty, otherwise returns `false`.
         */
        [[nodiscard]] bool has_value() const noexcept;
        /**
         * @brief Queries the contained type.
         * @returns The `type_info` of the contained value if instance is non-empty, otherwise `typeid(void)`.
         */
        [[nodiscard]] const std::type_info& type() const noexcept;

    private:
    // Friend Declarations
        template<std::size_t M> friend class sized_any;
        template<class T, std::size_t M>
        friend const T* any_cast(const sized_any<M>* operand) noexcept;
        template<class T, std::size_t M>
        friend T* any_cast(sized_any<M>* operand) noexcept;

    // Member variables
        const sized_any_detail::ITypeInfo* info;
        std::array<char, N> buff;
    };

    /**
     * @brief Performs type-safe access to the contained object.
     * @tparam T The type to which the contained object should be cast.
     * @tparam N The size of the buffer used by `operand`.
     * @param operand The `kmillet::sized_any<N>` to access.
     * @exception `std::bad_any_cast` if the `typeid` of the requested `T` does not match that of the contents of `operand`.
     * @return A reference to the object contained in `operand`, casted to type `T`.
     * @details The program is ill-formed if `std::is_constructible_v<T, const std::remove_cvref_t<T>&>` is `false`.
     * Throws `std::bad_any_cast` if the `typeid` of the requested `T` does not match that of the contents of `operand`.
     * Otherwise, returns `static_cast<T>(*any_cast<std::remove_cvref_t<T>>(&operand))`.
     */
    template<class T, std::size_t N>
    T any_cast(const sized_any<N>& operand);
    /**
     * @brief Performs type-safe access to the contained object.
     * @tparam T The type to which the contained object should be cast.
     * @tparam N The size of the buffer used by `operand`.
     * @param operand The `kmillet::sized_any<N>` to access.
     * @exception `std::bad_any_cast` if the `typeid` of the requested `T` does not match that of the contents of `operand`.
     * @return A reference to the object contained in `operand`, casted to type `T`.
     * @details The program is ill-formed if `std::is_constructible_v<T, std::remove_cvref_t<T>&>` is `false`.
     * Throws `std::bad_any_cast` if the `typeid` of the requested `T` does not match that of the contents of `operand`.
     * Otherwise, returns `static_cast<T>(*any_cast<std::remove_cvref_t<T>>(&operand))`.
     */
    template<class T, std::size_t N>
    T any_cast(sized_any<N>& operand);
    /**
     * @brief Performs type-safe access to the contained object.
     * @tparam T The type to which the contained object should be cast.
     * @tparam N The size of the buffer used by `operand`.
     * @param operand The `kmillet::sized_any<N>` to access.
     * @exception `std::bad_any_cast` if the `typeid` of the requested `T` does not match that of the contents of `operand`.
     * @return A reference to the object contained in `operand`, move-casted to type `T`.
     * @details The program is ill-formed if `std::is_constructible_v<T, std::remove_cvref_t<T>>` is `false`.
     * Throws `std::bad_any_cast` if the `typeid` of the requested `T` does not match that of the contents of `operand`.
     * Otherwise, returns `static_cast<T>(std::move(*any_cast<std::remove_cvref_t<T>>(&operand)))`.
     */
    template<class T, std::size_t N>
    T any_cast(sized_any<N>&& operand);
    /**
     * @brief Performs type-safe access to the contained object.
     * @tparam T The type to which the contained object should be cast.
     * @tparam N The size of the buffer used by `operand`.
     * @param operand The pointer to the `kmillet::sized_any<N>` to access.
     * @return A pointer to the object contained in `operand`, casted to type `const T*` if `operand` is not a null pointer and the `typeid` of the
     * requested `T` matches that of the contents of `operand`; otherwise returns a null pointer.
     * @details The program is ill-formed if `std::is_void_v<T>` is `true`.
     * If `operand` is not a null pointer and the `typeid` of the requested `T` matches that of the contents of `operand`,
     * returns a pointer to the value contained by `operand`; otherwise returns a null pointer.
     */
    template<class T, std::size_t N>
    const T* any_cast(const sized_any<N>* operand) noexcept;
    /**
     * @brief Performs type-safe access to the contained object.
     * @tparam T The type to which the contained object should be cast.
     * @tparam N The size of the buffer used by `operand`.
     * @param operand The pointer to the `kmillet::sized_any<N>` to access.
     * @return A pointer to the object contained in `operand`, casted to type `T*` if `operand` is not a null pointer and the `typeid` of the
     * requested `T` matches that of the contents of `operand`; otherwise returns a null pointer.
     * @details The program is ill-formed if `std::is_void_v<T>` is `true`.
     * If `operand` is not a null pointer and the `typeid` of the requested `T` matches that of the contents of `operand`,
     * returns a pointer to the value contained by `operand`; otherwise returns a null pointer.
     */
    template<class T, std::size_t N>
    T* any_cast(sized_any<N>* operand) noexcept;

    /**
     * @brief Constructs a `kmillet::sized_any<N>` object containing an object of type `T`, passing the provided arguments to `T`'s constructor.
     * @tparam N The size of the buffer used for in-place storage.
     * @tparam T The type of the value to be stored.
     * @tparam Args The types of the arguments to be forwarded to the constructor of `T`.
     * @param args The arguments to be forwarded to the constructor of `T`.
     * @return A `kmillet::sized_any<N>` object containing an object of type `T`, constructed with the provided arguments.
     * @details Equivalent to `return kmillet::sized_any<N>(std::in_place_type<T>, std::forward<Args>(args)...);`
     */
    template<std::size_t N, class T, class... Args>
    sized_any<N> make_sized_any(Args&&... args) noexcept(noexcept(sized_any<N>{std::in_place_type<T>, std::forward<Args>(args)...}));
    /**
     * @brief Constructs a `kmillet::sized_any<N>` object containing an object of type `T`, passing the provided arguments to `T`'s constructor.
     * @tparam N The size of the buffer used for in-place storage.
     * @tparam T The type of the value to be stored.
     * @tparam Args The types of the arguments to be forwarded to the constructor of `T`.
     * @param il The initializer list to be used for constructing the object.
     * @param args The arguments to be forwarded to the constructor of `T`.
     * @return A `kmillet::sized_any<N>` object containing an object of type `T`, constructed with the provided arguments.
     * @details Equivalent to `return kmillet::sized_any<N>(std::in_place_type<T>, il, std::forward<Args>(args)...);`
     */
    template<std::size_t N, class T, class U, class... Args>
    sized_any<N> make_sized_any(std::initializer_list<U> il, Args&&... args) noexcept(noexcept(sized_any<N>{std::in_place_type<T>, il, std::forward<Args>(args)...}));

    /**
     * @brief Constructs a `kmillet::sized_any` object containing an object of type `T`, passing the provided arguments to `T`'s constructor.
     * The capacity of the constructed and returned `kmillet::sized_any` is exactly big enough to hold the `T` instance.
     * @tparam T The type of the value to be stored.
     * @tparam Args The types of the arguments to be forwarded to the constructor of `T`.
     * @param args The arguments to be forwarded to the constructor of `T`.
     * @return A `kmillet::sized_any<sizeof(std::decay_t<T>)>` object containing an object of type `T`, constructed with the provided arguments.
     * @details Equivalent to `return kmillet::sized_any<sizeof(std::decay_t<T>)>(std::in_place_type<T>, std::forward<Args>(args)...);`
     */
    template<class T, class... Args>
    sized_any<sizeof(std::decay_t<T>)> make_sized_any(Args&&... args) noexcept(noexcept(make_sized_any<sizeof(std::decay_t<T>), T>(std::forward<Args>(args)...)));
    /**
     * @brief Constructs a `kmillet::sized_any` object containing an object of type `T`, passing the provided arguments to `T`'s constructor.
     * The capacity of the constructed and returned `kmillet::sized_any` is exactly big enough to hold the `T` instance.
     * @tparam T The type of the value to be stored.
     * @tparam U The type of the elements in the initializer list.
     * @tparam Args The types of the arguments to be forwarded to the constructor of `T`.
     * @param il The initializer list to be used for constructing the object.
     * @param args The arguments to be forwarded to the constructor of `T`.
     * @return A `kmillet::sized_any<sizeof(std::decay_t<T>)>` object containing an object of type `T`, constructed with the provided arguments.
     * @details Equivalent to `return kmillet::sized_any<sizeof(std::decay_t<T>)>(std::in_place_type<T>, il, std::forward<Args>(args)...);`
     */
    template<class T, class U, class... Args>
    sized_any<sizeof(std::decay_t<T>)> make_sized_any(std::initializer_list<U> il, Args&&... args) noexcept(noexcept(make_sized_any<sizeof(std::decay_t<T>), T>(il, std::forward<Args>(args)...)));

    /**
     * @brief An alias for a `kmillet::sized_any<N>` where `N` is equal to the internal buffer size of `std::any`.
     * For most practical purposes, it is intended to be identical to `std::any`.
     * @details Intended to be used as a direct replacement for `std::any`, providing a more efficient implementation with better control over memory usage
     * and compatibility with other specializations of `kmillet::sized_any`.
     */
    using any = sized_any<sizeof(std::any)-sizeof(void*)>;
    /**
     * @brief An alias for `kmillet::make_sized_any<N, T>` where `N` is equal to the internal buffer size of `std::any`.
     * For most practical purposes, it is intended to be identical to `std::make_any` aside from the return type.
     * @tparam T The type of the value to be stored.
     * @tparam Args The types of the arguments to be forwarded to the constructor of `T`.
     * @param args The arguments to be forwarded to the constructor of `T`.
     * @return A `kmillet::any` object containing an object of type `T`, constructed with the provided arguments.
     * @details Intended to be used as a direct replacement for `std::make_any` where the desired return type is a `kmillet::any` instead of a `std::any`.
     * Equivalent to `return kmillet::sized_any<kmillet::any::capacity()>(std::in_place_type<T>, std::forward<Args>(args)...);`
     */
    template<class T, class... Args>
    any make_any(Args&&... args) noexcept(noexcept(make_sized_any<any::capacity(), T>(std::forward<Args>(args)...)));
    /**
     * @brief `kmillet::make_any<T>` is an alias for `kmillet::make_sized_any<N, T>` where `N` is equal to the internal buffer size of `std::any`.
     * @tparam T The type of the value to be stored.
     * @param U The type of the elements in the initializer list.
     * @tparam Args The types of the arguments to be forwarded to the constructor of `T`.
     * @param il The initializer list to be used for constructing the object.
     * @param args The arguments to be forwarded to the constructor of `T`.
     * @return A `kmillet::any` object containing an object of type `T`, constructed with the provided arguments.
     * @details Intended to be used as a direct replacement for `std::make_any` where the desired return type is a `kmillet::any` instead of a `std::any`.
     * Equivalent to `return kmillet::sized_any<kmillet::any::capacity()>(std::in_place_type<T>, il, std::forward<Args>(args)...);`
     */
    template<class T, class U, class... Args>
    any make_any(std::initializer_list<U> il, Args&&... args) noexcept(noexcept(make_sized_any<any::capacity(), T>(il, std::forward<Args>(args)...)));
}



// ----------------------------------------------------------------------------
// Implementation details below this point.
// ----------------------------------------------------------------------------

struct kmillet::sized_any_detail::ITypeInfo
{
    virtual constexpr const std::type_info& type() const noexcept = 0;
    virtual constexpr std::size_t size() const noexcept = 0;
    virtual constexpr bool needsAlloc(std::size_t cap) const noexcept = 0;
    virtual void copy(const void* from, char* to, std::size_t fromCap, std::size_t toCap) const = 0;
    virtual void move(void* from, char* to, std::size_t fromCap, std::size_t toCap) const = 0;
    virtual void destroy(char* buff, std::size_t cap) const noexcept = 0;
    virtual void del(void* buff, std::size_t cap) const noexcept = 0;
};

template<>
struct kmillet::sized_any_detail::TypeInfo<void> final : kmillet::sized_any_detail::ITypeInfo
{
    constexpr const std::type_info& type() const noexcept override { return typeid(void); }
    constexpr std::size_t size() const noexcept override { return 0; }
    constexpr bool needsAlloc(std::size_t) const noexcept override { return false; }
    void copy(const void*, char*, std::size_t, std::size_t) const override{}
    void move(void*, char*, std::size_t, std::size_t) const override {}
    void destroy(char*, std::size_t) const noexcept override {}
    void del(void*, std::size_t) const noexcept override {}
};

template<class T>
struct kmillet::sized_any_detail::TypeInfo final : kmillet::sized_any_detail::ITypeInfo
{
    constexpr const std::type_info& type() const noexcept override { return typeid(T); }
    constexpr std::size_t size() const noexcept override { return sizeof(T); }
    constexpr bool needsAlloc(std::size_t cap) const noexcept override
    {
        return sizeof(T) > cap || !std::is_nothrow_move_constructible_v<T>;
    }
    const T& getValueForCopying(const void* from, std::size_t fromCap) const
    {
        if (needsAlloc(fromCap))
        {
            return **reinterpret_cast<const T**>(from);
        }
        else return *reinterpret_cast<const T*>(from);
    }
    T& getValueForMoving(void* from, std::size_t fromCap) const
    {
        if (needsAlloc(fromCap))
        {
            return **reinterpret_cast<T**>(from);
        }
        else return *reinterpret_cast<T*>(from);
    }
    void copy(const void* from, char* to, std::size_t fromCap, std::size_t toCap) const override
    {
        if (needsAlloc(toCap))
        {
            *reinterpret_cast<const void**>(to) = new T(getValueForCopying(from, fromCap));
        }
        else new (to) T(getValueForCopying(from, fromCap));
    }
    void move(void* from, char* to, std::size_t fromCap, std::size_t toCap) const override
    {
        if (needsAlloc(toCap))
        {
            if (needsAlloc(fromCap))
            {
                *reinterpret_cast<void**>(to) = *reinterpret_cast<void**>(from);
                return;
            }
            else *reinterpret_cast<void**>(to) = new T(std::move(*reinterpret_cast<T*>(from)));
        }
        else new (to) T(std::move(getValueForMoving(from, fromCap)));
        del(from, fromCap);
    }
    void destroy(char* buff, std::size_t cap) const noexcept override
    {
        if (needsAlloc(cap))
        {
            (*reinterpret_cast<T**>(buff))->~T();
        }
        else reinterpret_cast<T*>(buff)->~T();
    }
    void del(void* buff, std::size_t cap) const noexcept override
    {
        if (needsAlloc(cap))
        {
            delete *reinterpret_cast<T**>(buff);
        }
        else reinterpret_cast<T*>(buff)->~T();
    }
};

template <std::size_t N>
inline constexpr kmillet::sized_any<N>::sized_any() noexcept
    : info(&(kmillet::sized_any_detail::info<void>))
{}
template <std::size_t N>
inline kmillet::sized_any<N>::sized_any(const kmillet::sized_any<N>& other)
    : info(other.info)
{
    info->copy(other.buff.data(), buff.data(), N, N);
}
template <std::size_t N>
template<std::size_t M>
inline kmillet::sized_any<N>::sized_any(const kmillet::sized_any<M>& other)
    : info(other.info)
{
    info->copy(other.buff.data(), buff.data(), M, N);
}
template <std::size_t N>
inline kmillet::sized_any<N>::sized_any(kmillet::sized_any<N>&& other) noexcept
    : info(other.info)
{
    info->move(other.buff.data(), buff.data(), N, N);
    other.info = &(kmillet::sized_any_detail::info<void>);
}
template <std::size_t N>
template<std::size_t M>
inline kmillet::sized_any<N>::sized_any(kmillet::sized_any<M>&& other) noexcept(M < N)
    : info(other.info)
{
    info->move(other.buff.data(), buff.data(), M, N);
    other.info = &(kmillet::sized_any_detail::info<void>);
}
template <std::size_t N>
template <class ValueType>
requires(std::conjunction_v<std::negation<kmillet::sized_any_detail::is_sized_any<std::decay_t<ValueType>>>, std::negation<kmillet::sized_any_detail::is_in_place_type<std::decay_t<ValueType>>>, std::is_copy_constructible<std::decay_t<ValueType>>>)
inline kmillet::sized_any<N>::sized_any(ValueType&& value) noexcept(std::is_nothrow_constructible_v<std::decay_t<ValueType>, ValueType> && kmillet::sized_any_optimized<ValueType, N>)
    : info(&(kmillet::sized_any_detail::info<std::decay_t<ValueType>>))
{
    if constexpr (kmillet::sized_any_detail::info<std::decay_t<ValueType>>.needsAlloc(N))
    {
        *reinterpret_cast<void**>(buff.data()) = new std::decay_t<ValueType>(std::forward<ValueType>(value));
    }
    else new (buff.data()) std::decay_t<ValueType>(std::forward<ValueType>(value));
}
template <std::size_t N>
template <class ValueType, class... Args>
requires(std::copy_constructible<std::decay_t<ValueType>> && std::constructible_from<std::decay_t<ValueType>, Args...>)
inline kmillet::sized_any<N>::sized_any(std::in_place_type_t<ValueType>, Args&&... args) noexcept(std::is_nothrow_constructible_v<std::decay_t<ValueType>, Args...> && kmillet::sized_any_optimized<ValueType, N>)
    : info(&(kmillet::sized_any_detail::info<std::decay_t<ValueType>>))
{
    if constexpr (kmillet::sized_any_detail::info<std::decay_t<ValueType>>.needsAlloc(N))
    {
        *reinterpret_cast<void**>(buff.data()) = new std::decay_t<ValueType>(std::forward<Args>(args)...);
    }
    else new (buff.data()) std::decay_t<ValueType>(std::forward<Args>(args)...);
}
template <std::size_t N>
template <class ValueType, class U, class... Args>
requires(std::copy_constructible<std::decay_t<ValueType>> && std::constructible_from<std::decay_t<ValueType>, std::initializer_list<U>&, Args...>)
inline kmillet::sized_any<N>::sized_any(std::in_place_type_t<ValueType>, std::initializer_list<U> il, Args&&... args) noexcept(std::is_nothrow_constructible_v<std::decay_t<ValueType>, std::initializer_list<U>&, Args...> && kmillet::sized_any_optimized<ValueType, N>)
    : info(&(kmillet::sized_any_detail::info<std::decay_t<ValueType>>))
{
    if constexpr (kmillet::sized_any_detail::info<std::decay_t<ValueType>>.needsAlloc(N))
    {
        *reinterpret_cast<void**>(buff.data()) = new std::decay_t<ValueType>(il, std::forward<Args>(args)...);
    }
    else new (buff.data()) std::decay_t<ValueType>(il, std::forward<Args>(args)...);
}

template <std::size_t N>
inline kmillet::sized_any<N>::~sized_any()
{
    reset();
}

template <std::size_t N>
inline kmillet::sized_any<N>& kmillet::sized_any<N>::operator=(const kmillet::sized_any<N>& rhs)
{
    if (&rhs == this) return *this;
    kmillet::sized_any<N>(rhs).swap(*this);
    return *this;
}
template <std::size_t N>
template <std::size_t M>
inline kmillet::sized_any<N>& kmillet::sized_any<N>::operator=(const kmillet::sized_any<M>& rhs)
{
    kmillet::sized_any<N>(rhs).swap(*this);
    return *this;
}
template <std::size_t N>
inline kmillet::sized_any<N>& kmillet::sized_any<N>::operator=(kmillet::sized_any<N>&& rhs) noexcept
{
    if (&rhs == this) return *this;
    kmillet::sized_any<N>(std::move(rhs)).swap(*this);
    return *this;
}
template <std::size_t N>
template <std::size_t M>
inline kmillet::sized_any<N>& kmillet::sized_any<N>::operator=(kmillet::sized_any<M>&& rhs) noexcept(M < N)
{
    kmillet::sized_any<N>(std::move(rhs)).swap(*this);
    return *this;
}
template <std::size_t N>
template <class ValueType>
requires(std::conjunction_v<std::negation<kmillet::sized_any_detail::is_sized_any<std::decay_t<ValueType>>>, std::is_copy_constructible<std::decay_t<ValueType>>>)
inline kmillet::sized_any<N>& kmillet::sized_any<N>::operator=(ValueType&& rhs) noexcept(noexcept(kmillet::sized_any<N>{std::forward<ValueType>(rhs)}))
{
    kmillet::sized_any<N>(std::forward<ValueType>(rhs)).swap(*this);
    return *this;
}

template <std::size_t N>
template <class ValueType, class... Args>
requires(std::copy_constructible<std::decay_t<ValueType>> && std::constructible_from<std::decay_t<ValueType>, Args...>)
inline std::decay_t<ValueType>& kmillet::sized_any<N>::emplace(Args&&... args) noexcept(noexcept(kmillet::sized_any<N>{std::in_place_type<ValueType>, std::forward<Args>(args)...}))
{
    if constexpr (kmillet::sized_any_detail::info<std::decay_t<ValueType>>.needsAlloc(N))
    {
        if (info->needsAlloc(N) && info->size() == kmillet::sized_any_detail::info<std::decay_t<ValueType>>.size())
        {
            info->destroy(buff.data(), N);
            new (*reinterpret_cast<void**>(buff.data())) std::decay_t<ValueType>(std::forward<Args>(args)...);
        }
        else
        {
            info->del(buff.data(), N);
            *reinterpret_cast<void**>(buff.data()) = new std::decay_t<ValueType>(std::forward<Args>(args)...);
        }
        info = &(kmillet::sized_any_detail::info<std::decay_t<ValueType>>);
        return **reinterpret_cast<std::decay_t<ValueType>**>(buff.data());
    }
    else
    {
        info->del(buff.data(), N);
        new (buff.data()) std::decay_t<ValueType>(std::forward<Args>(args)...);
        info = &(kmillet::sized_any_detail::info<std::decay_t<ValueType>>);
        return *reinterpret_cast<std::decay_t<ValueType>*>(buff.data());
    }
}
template <std::size_t N>
template<class ValueType, class U, class... Args>
requires(std::copy_constructible<std::decay_t<ValueType>> && std::constructible_from<std::decay_t<ValueType>, std::initializer_list<U>&, Args...>)
inline std::decay_t<ValueType>& kmillet::sized_any<N>::emplace(std::initializer_list<U> il, Args&&... args) noexcept(noexcept(kmillet::sized_any<N>{std::in_place_type<ValueType>, il, std::forward<Args>(args)...}))
{
    if constexpr (kmillet::sized_any_detail::info<std::decay_t<ValueType>>.needsAlloc(N))
    {
        if (info->needsAlloc(N) && info->size() == kmillet::sized_any_detail::info<std::decay_t<ValueType>>.size())
        {
            info->destroy(buff.data(), N);
            new (*reinterpret_cast<void**>(buff.data())) std::decay_t<ValueType>(il, std::forward<Args>(args)...);
        }
        else
        {
            info->del(buff.data(), N);
            *reinterpret_cast<void**>(buff.data()) = new std::decay_t<ValueType>(il, std::forward<Args>(args)...);
        }
        info = &(kmillet::sized_any_detail::info<std::decay_t<ValueType>>);
        return **reinterpret_cast<std::decay_t<ValueType>**>(buff.data());
    }
    else
    {
        info->del(buff.data(), N);
        new (buff.data()) std::decay_t<ValueType>(il, std::forward<Args>(args)...);
        info = &(kmillet::sized_any_detail::info<std::decay_t<ValueType>>);
        return *reinterpret_cast<std::decay_t<ValueType>*>(buff.data());
    }
}

template <std::size_t N>
inline void kmillet::sized_any<N>::reset() noexcept
{
    if (info == &(kmillet::sized_any_detail::info<void>)) return;
    info->del(buff.data(), N);
    info = &(kmillet::sized_any_detail::info<void>);
}
template <std::size_t N>
inline void kmillet::sized_any<N>::swap(kmillet::sized_any<N>& other) noexcept
{
    if (&other == this) return; 
    std::array<char, N> tmp;
    other.info->move(other.buff.data(), tmp.data(), N, N);
    info->move(buff.data(), other.buff.data(), N, N);
    info->move(tmp.data(), buff.data(), N, N);
    if (info != other.info) std::swap(info, other.info);
}
template <std::size_t N>
template <std::size_t M>
inline void kmillet::sized_any<N>::swap(kmillet::sized_any<M>& other)
{
    if constexpr (M < N)
    {
        std::array<char, M> tmp;
        info->move(buff.data(), tmp.data(), N, M);
        other.info->move(other.buff.data(), buff.data(), M, N);
        info->move(tmp.data(), other.buff.data(), M, M);
    }
    else
    {
        std::array<char, N> tmp;
        other.info->move(other.buff.data(), tmp.data(), M, N);
        info->move(buff.data(), other.buff.data(), N, M);
        info->move(tmp.data(), buff.data(), N, N);
    }
    if (info != other.info) std::swap(info, other.info);
}

template <std::size_t N>
inline bool kmillet::sized_any<N>::has_value() const noexcept
{
    return info != &(kmillet::sized_any_detail::info<void>);
}
template <std::size_t N>
inline const std::type_info& kmillet::sized_any<N>::type() const noexcept
{
    return info->type();
}

template <class T, std::size_t N>
T kmillet::any_cast(const kmillet::sized_any<N>& operand)
{
    if (auto* casted = any_cast<std::remove_cvref_t<T>>(&operand)) return static_cast<T>(*casted);
    throw std::bad_any_cast{};
}
template <class T, std::size_t N>
T kmillet::any_cast(kmillet::sized_any<N>& operand)
{
    if (auto* casted = any_cast<std::remove_cvref_t<T>>(&operand)) return static_cast<T>(*casted);
    throw std::bad_any_cast{};
}
template <class T, std::size_t N>
T kmillet::any_cast(kmillet::sized_any<N>&& operand)
{
    if (auto* casted = any_cast<std::remove_cvref_t<T>>(&operand)) return static_cast<T>(std::move(*casted));
    throw std::bad_any_cast{};
}
template <class T, std::size_t N>
const T* kmillet::any_cast(const kmillet::sized_any<N>* operand) noexcept
{
    if (!operand || operand->info != &(kmillet::sized_any_detail::info<std::decay_t<T>>)) return nullptr;
    if constexpr (kmillet::sized_any_detail::info<std::decay_t<T>>.needsAlloc(N))
    {
        return *reinterpret_cast<const T**>(operand->buff.data());
    }
    else return reinterpret_cast<const T*>(operand->buff.data());
}
template <class T, std::size_t N>
T* kmillet::any_cast(kmillet::sized_any<N>* operand) noexcept
{
    if (!operand || operand->info != &(kmillet::sized_any_detail::info<std::decay_t<T>>)) return nullptr;
    if constexpr (kmillet::sized_any_detail::info<std::decay_t<T>>.needsAlloc(N))
    {
        return *reinterpret_cast<T**>(operand->buff.data());
    }
    else return reinterpret_cast<T*>(operand->buff.data());
}

template <std::size_t N, class T, class... Args>
kmillet::sized_any<N> kmillet::make_sized_any(Args&&... args) noexcept(noexcept(kmillet::sized_any<N>{std::in_place_type<T>, std::forward<Args>(args)...}))
{
    return kmillet::sized_any<N>{std::in_place_type<T>, std::forward<Args>(args)...};
}
template <std::size_t N, class T, class U, class... Args>
kmillet::sized_any<N> kmillet::make_sized_any(std::initializer_list<U> il, Args&&... args) noexcept(noexcept(kmillet::sized_any<N>{std::in_place_type<T>, il, std::forward<Args>(args)...}))
{
    return kmillet::sized_any<N>{std::in_place_type<T>, il, std::forward<Args>(args)...};
}
template<class T, class... Args>
kmillet::sized_any<sizeof(std::decay_t<T>)> kmillet::make_sized_any(Args&&... args) noexcept(noexcept(kmillet::make_sized_any<sizeof(std::decay_t<T>), T>(std::forward<Args>(args)...)))
{
    return kmillet::make_sized_any<sizeof(std::decay_t<T>), T>(std::forward<Args>(args)...);
}
template<class T, class U, class... Args>
kmillet::sized_any<sizeof(std::decay_t<T>)> kmillet::make_sized_any(std::initializer_list<U> il, Args&&... args) noexcept(noexcept(kmillet::make_sized_any<sizeof(std::decay_t<T>), T>(il, std::forward<Args>(args)...)))
{
    return kmillet::make_sized_any<sizeof(std::decay_t<T>), T>(il, std::forward<Args>(args)...);
}
template <class T, class... Args>
kmillet::any kmillet::make_any(Args &&...args) noexcept(noexcept(kmillet::make_sized_any<kmillet::any::capacity(), T>(std::forward<Args>(args)...)))
{
    return kmillet::make_sized_any<kmillet::any::capacity(), T>(std::forward<Args>(args)...);
}
template <class T, class U, class... Args>
kmillet::any kmillet::make_any(std::initializer_list<U> il, Args &&...args) noexcept(noexcept(kmillet::make_sized_any<kmillet::any::capacity(), T>(il, std::forward<Args>(args)...)))
{
    return kmillet::make_sized_any<kmillet::any::capacity(), T>(il, std::forward<Args>(args)...);
}