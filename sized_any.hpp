#pragma once

#include <any>
#include <array>
#include <concepts>
#include <cstddef>

namespace kmillet
{
    template <std::size_t N>
    class sized_any;

    using any = sized_any<sizeof(std::any)-sizeof(void*)>;

    using std::in_place_type_t;
    using std::in_place_type;
    using std::bad_any_cast;

    template<class T, std::size_t N>
    T any_cast(const sized_any<N>& operand);
    template<class T, std::size_t N>
    T any_cast(sized_any<N>& operand);
    template<class T, std::size_t N>
    T any_cast(sized_any<N>&& operand);
    template<class T, std::size_t N>
    const T* any_cast(const sized_any<N>* operand) noexcept;
    template<class T, std::size_t N>
    T* any_cast(sized_any<N>* operand) noexcept;

    namespace sized_any_detail
    {
        template<class T> struct is_sized_any : std::false_type {};
        template<std::size_t N> struct is_sized_any<sized_any<N>> : std::true_type {};

        template<class T> struct is_in_place_type : std::false_type {};
        template<class T> struct is_in_place_type<in_place_type_t<T>> : std::true_type {};

        struct ITypeInfo;
        template<class T> struct TypeInfo;
        template<class T> inline constexpr TypeInfo<T> info{};
    }

    template<std::size_t N, class T, class... Args>
    sized_any<N> make_sized_any(Args&&... args) noexcept(noexcept(sized_any<N>{in_place_type<T>, std::forward<Args>(args)...}));
    template<std::size_t N, class T, class U, class... Args>
    sized_any<N> make_sized_any(std::initializer_list<U> il, Args&&... args) noexcept(noexcept(sized_any<N>{in_place_type<T>, il, std::forward<Args>(args)...}));

    template<class T, class... Args>
    sized_any<sizeof(std::decay_t<T>)> make_sized_any(Args&&... args) noexcept(noexcept(make_sized_any<sizeof(std::decay_t<T>), T>(std::forward<Args>(args)...)));
    template<class T, class U, class... Args>
    sized_any<sizeof(std::decay_t<T>)> make_sized_any(std::initializer_list<U> il, Args&&... args) noexcept(noexcept(make_sized_any<sizeof(std::decay_t<T>), T>(il, std::forward<Args>(args)...)));

    // The concept ```sized_any_optimized<T, N>``` is satisfied if and only if a ```sized_any<N>``` will not
    // perform dynamic allocation to hold an object of type ```std::decay_t<T>```. This means the that both of
    // the following conditions hold true:
    // 1. ```sizeof(std::decay_t<T>) <= N```.
    // 2. Move-constructing ```std::decay_t<T>``` is noexcept.
    template<class T, std::size_t N>
    concept sized_any_optimized = !sized_any_detail::info<std::decay_t<T>>.needsAlloc(N);

    template<std::size_t N>
    class sized_any
    {
        static_assert(N >= sizeof(void*));
    public:
        // Constructs an empty object.
        constexpr sized_any() noexcept;
        // Copies the content of other into a new instance.
        // No dynamic allocation will occur if the content of other are smaller than ```N``` bytes in size.
        sized_any(const sized_any& other);
        // Copies the content of other into a new instance
        // No dynamic allocation will occur if the content of other are smaller than ```N``` bytes in size.
        template<std::size_t M>
        sized_any(const sized_any<M>& other);
        // Moves the content of other into a new instance.
        // No dynamic allocation will occur if move-constructing the content of other is noexcept.
        sized_any(sized_any&& other) noexcept;
        // Moves the content of other into a new instance.
        // No dynamic allocation will occur if ```M <= N``` and move-constructing the content of other is noexcept.
        template<std::size_t M>
        sized_any(sized_any<M>&& other) noexcept(M <= N);
        // Constructs an object with initial content an object of type ```std::decay_t<ValueType>```, direct-initialized from ```std::forward<ValueType>(value)```.
        // Requires that ```std::decay_t<ValueType>``` is not a sized_any nor a specialization of ```in_place_type_t```, and is copy-constructible.
        // Noexcept so long as constructing ```std::decay_t<ValueType>``` from ```std::forward<ValueType>(value)``` is noexcept and sized_any_optimized<ValueType, N> is satisfied.
        template<class ValueType>
        requires(std::conjunction_v<std::negation<sized_any_detail::is_sized_any<std::decay_t<ValueType>>>, std::negation<sized_any_detail::is_in_place_type<std::decay_t<ValueType>>>, std::is_copy_constructible<std::decay_t<ValueType>>>)
        sized_any(ValueType&& value) noexcept(std::is_nothrow_constructible_v<std::decay_t<ValueType>, ValueType> && sized_any_optimized<ValueType, N>);
        // Constructs an object with initial content an object of type ```std::decay_t<ValueType>```, direct-non-list-initialized from ```std::forward<Args>(args)...```.
        // Requires that ```std::decay_t<ValueType>``` is copy-constructible.
        // Noexcept so long as constructing ```std::decay_t<ValueType>``` direct-non-list-initialized from ```std::forward<Args>(args)...``` is noexcept, ```sizeof(std::decay_t<ValueType>) <= N```,
        // and move-constructing ```std::decay_t<ValueType>``` is noexcept.
        template<class ValueType, class... Args>
        requires(std::copy_constructible<std::decay_t<ValueType>> && std::constructible_from<std::decay_t<ValueType>, Args...>)
        explicit sized_any(in_place_type_t<ValueType>, Args&&... args) noexcept(std::is_nothrow_constructible_v<std::decay_t<ValueType>, Args...> && sized_any_optimized<ValueType, N>);
        // Constructs an object with initial content an object of type ```std::decay_t<ValueType>```, direct-non-list-initialized from ```il, std::forward<Args>(args)...```.
        // Requires that ```std::decay_t<ValueType>``` is copy-constructible.
        // Noexcept so long as constructing ```std::decay_t<ValueType>``` direct-non-list-initialized from ```il, std::forward<Args>(args)...``` is noexcept and sized_any_optimized<ValueType, N> is satisfied.
        template<class ValueType, class U, class... Args>
        requires(std::copy_constructible<std::decay_t<ValueType>> && std::constructible_from<std::decay_t<ValueType>, std::initializer_list<U>&, Args...>)
        explicit sized_any(in_place_type_t<ValueType>, std::initializer_list<U> il, Args&&... args) noexcept(std::is_nothrow_constructible_v<std::decay_t<ValueType>, std::initializer_list<U>&, Args...> && sized_any_optimized<ValueType, N>);

        // Destroys the contained object, if any, as if by a call to reset().
        ~sized_any();

        sized_any& operator=(const sized_any& rhs);
        template<std::size_t M>
        sized_any& operator=(const sized_any<M>& rhs);
        sized_any& operator=(sized_any&& rhs) noexcept;
        template<std::size_t M>
        sized_any& operator=(sized_any<M>&& rhs) noexcept(M <= N);
        template<class ValueType>
        requires(std::conjunction_v<std::negation<sized_any_detail::is_sized_any<std::decay_t<ValueType>>>, std::is_copy_constructible<std::decay_t<ValueType>>>)
        sized_any& operator=(ValueType&& rhs) noexcept(noexcept(sized_any{std::forward<ValueType>(rhs)}));

        template<class ValueType, class... Args>
        requires(std::copy_constructible<std::decay_t<ValueType>> && std::constructible_from<std::decay_t<ValueType>, Args...>)
        std::decay_t<ValueType>& emplace(Args&&... args) noexcept(noexcept(sized_any{in_place_type<ValueType>, std::forward<Args>(args)...}));
        template<class ValueType, class U, class... Args>
        requires(std::copy_constructible<std::decay_t<ValueType>> && std::constructible_from<std::decay_t<ValueType>, std::initializer_list<U>&, Args...>)
        std::decay_t<ValueType>& emplace(std::initializer_list<U> il, Args&&... args) noexcept(noexcept(sized_any{in_place_type<ValueType>, il, std::forward<Args>(args)...}));

        void reset() noexcept;
        void swap(sized_any& other) noexcept;
        // template<std::size_t M>
        // void swap(sized_any<M>& other);

        bool has_value() const noexcept;
        const std::type_info& type() const noexcept;
    private:
        template<std::size_t M> friend class sized_any;
        template<class T, std::size_t M>
        friend const T* any_cast(const sized_any<M>* operand) noexcept;
        template<class T, std::size_t M>
        friend T* any_cast(sized_any<M>* operand) noexcept;
    private:
        const sized_any_detail::ITypeInfo* info;
        std::array<char, N> buff;
    };

    struct sized_any_detail::ITypeInfo
    {
        virtual const std::type_info& type() const noexcept = 0;
        virtual constexpr std::size_t size() const = 0;
        virtual constexpr bool needsAlloc(std::size_t cap) const = 0;
        virtual void copy(const void* from, char* to, std::size_t fromCap, std::size_t toCap) const = 0;
        virtual void move(void* from, char* to, std::size_t fromCap, std::size_t toCap) const = 0;
        virtual void replCopy(const void* from, char* to, const ITypeInfo& toInfo, std::size_t fromCap, std::size_t toCap) const = 0;
        virtual void replMove(void* from, char* to, const ITypeInfo& toInfo, std::size_t fromCap, std::size_t toCap) const = 0;
        virtual void destroy(char* buff, std::size_t cap) const = 0;
        virtual void del(void* buff, std::size_t cap) const = 0;
    };

    template<>
    struct sized_any_detail::TypeInfo<void> final : sized_any_detail::ITypeInfo
    {
        const std::type_info& type() const noexcept override { return typeid(void); }
        constexpr std::size_t size() const override { return 0; }
        constexpr bool needsAlloc(std::size_t) const override { return false; }
        void copy(const void*, char*, std::size_t, std::size_t) const override{}
        void move(void*, char*, std::size_t, std::size_t) const override {}
        void replCopy(const void*, char*, const ITypeInfo&, std::size_t, std::size_t) const override {}
        void replMove(void*, char*, const ITypeInfo&, std::size_t, std::size_t) const override {}
        void destroy(char*, std::size_t) const override {}
        void del(void*, std::size_t) const override {}
    };

    template<class T>
    struct sized_any_detail::TypeInfo final : sized_any_detail::ITypeInfo
    {
        const std::type_info& type() const noexcept override { return typeid(T); }
        constexpr std::size_t size() const override { return sizeof(T); }
        constexpr bool needsAlloc(std::size_t cap) const override
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
        void replCopy(const void* from, char* to, const ITypeInfo& toInfo, std::size_t fromCap, std::size_t toCap) const override
        {
            if (needsAlloc(toCap))
            {
                if (toInfo.needsAlloc(toCap) && toInfo.size() == size())
                {
                    toInfo.destroy(to, toCap);
                    new (to) T(getValueForCopying(from, fromCap));
                }
                else
                {
                    toInfo.del(to, toCap);
                    *reinterpret_cast<void**>(to) = new T(getValueForCopying(from, fromCap));
                }
            }
            else
            {
                toInfo.del(to, toCap);
                new (to) T(getValueForCopying(from, fromCap));
            }
        }
        void replMove(void* from, char* to, const ITypeInfo& toInfo, std::size_t fromCap, std::size_t toCap) const override
        {
            if (needsAlloc(toCap))
            {
                if (needsAlloc(fromCap))
                {
                    toInfo.del(to, toCap);
                    *reinterpret_cast<void**>(to) = *reinterpret_cast<void**>(from);
                    return;
                }
                else if (toInfo.needsAlloc(toCap) && toInfo.size() == size())
                {
                    toInfo.destroy(to, toCap);
                    new (to) T(std::move(getValueForMoving(from, fromCap)));
                }
                else
                {
                    toInfo.del(to, toCap);
                    *reinterpret_cast<void**>(to) = new T(std::move(getValueForMoving(from, fromCap)));
                }
            }
            else
            {
                toInfo.del(to, toCap);
                new (to) T(std::move(getValueForMoving(from, fromCap)));
            }
            del(from, fromCap);
        }
        void destroy(char* buff, std::size_t cap) const override
        {
            if (needsAlloc(cap))
            {
                (*reinterpret_cast<T**>(buff))->~T();
            }
            else reinterpret_cast<T*>(buff)->~T();
        }
        void del(void* buff, std::size_t cap) const override
        {
            if (needsAlloc(cap))
            {
                delete *reinterpret_cast<T**>(buff);
            }
            else reinterpret_cast<T*>(buff)->~T();
        }
    };

    template <std::size_t N>
    inline constexpr sized_any<N>::sized_any() noexcept
        : info(&(sized_any_detail::info<void>))
    {}
    template <std::size_t N>
    inline sized_any<N>::sized_any(const sized_any& other)
        : info(other.info)
    {
        info->copy(other.buff.data(), buff.data(), N, N);
    }
    template <std::size_t N>
    template<std::size_t M>
    inline sized_any<N>::sized_any(const sized_any<M>& other)
        : info(other.info)
    {
        info->copy(other.buff.data(), buff.data(), M, N);
    }
    template <std::size_t N>
    inline sized_any<N>::sized_any(sized_any&& other) noexcept
        : info(other.info)
    {
        info->move(other.buff.data(), buff.data(), N, N);
        other.info = &(sized_any_detail::info<void>);
    }
    template <std::size_t N>
    template<std::size_t M>
    inline sized_any<N>::sized_any(sized_any<M>&& other) noexcept(M <= N)
        : info(other.info)
    {
        info->move(other.buff.data(), buff.data(), M, N);
        other.info = &(sized_any_detail::info<void>);
    }
    template <std::size_t N>
    template <class ValueType>
    requires(std::conjunction_v<std::negation<sized_any_detail::is_sized_any<std::decay_t<ValueType>>>, std::negation<sized_any_detail::is_in_place_type<std::decay_t<ValueType>>>, std::is_copy_constructible<std::decay_t<ValueType>>>)
    inline sized_any<N>::sized_any(ValueType&& value) noexcept(std::is_nothrow_constructible_v<std::decay_t<ValueType>, ValueType> && sized_any_optimized<ValueType, N>)
        : info(&(sized_any_detail::info<std::decay_t<ValueType>>))
    {
        if constexpr (sized_any_detail::info<std::decay_t<ValueType>>.needsAlloc(N))
        {
            *reinterpret_cast<void**>(buff.data()) = new std::decay_t<ValueType>(std::forward<ValueType>(value));
        }
        else new (buff.data()) std::decay_t<ValueType>(std::forward<ValueType>(value));
    }
    template <std::size_t N>
    template <class ValueType, class... Args>
    requires(std::copy_constructible<std::decay_t<ValueType>> && std::constructible_from<std::decay_t<ValueType>, Args...>)
    inline sized_any<N>::sized_any(in_place_type_t<ValueType>, Args&&... args) noexcept(std::is_nothrow_constructible_v<std::decay_t<ValueType>, Args...> && sized_any_optimized<ValueType, N>)
        : info(&(sized_any_detail::info<std::decay_t<ValueType>>))
    {
        if constexpr (sized_any_detail::info<std::decay_t<ValueType>>.needsAlloc(N))
        {
            *reinterpret_cast<void**>(buff.data()) = new std::decay_t<ValueType>(std::forward<Args>(args)...);
        }
        else new (buff.data()) std::decay_t<ValueType>(std::forward<Args>(args)...);
    }
    template <std::size_t N>
    template <class ValueType, class U, class... Args>
    requires(std::copy_constructible<std::decay_t<ValueType>> && std::constructible_from<std::decay_t<ValueType>, std::initializer_list<U>&, Args...>)
    inline sized_any<N>::sized_any(in_place_type_t<ValueType>, std::initializer_list<U> il, Args&&... args) noexcept(std::is_nothrow_constructible_v<std::decay_t<ValueType>, std::initializer_list<U>&, Args...> && sized_any_optimized<ValueType, N>)
        : info(&(sized_any_detail::info<std::decay_t<ValueType>>))
    {
        if constexpr (sized_any_detail::info<std::decay_t<ValueType>>.needsAlloc(N))
        {
            *reinterpret_cast<void**>(buff.data()) = new std::decay_t<ValueType>(il, std::forward<Args>(args)...);
        }
        else new (buff.data()) std::decay_t<ValueType>(il, std::forward<Args>(args)...);
    }

    template <std::size_t N>
    inline sized_any<N>::~sized_any()
    {
        info->del(buff.data(), N);
    }

    template <std::size_t N>
    inline sized_any<N>& sized_any<N>::operator=(const sized_any& rhs)
    {
        if (&rhs == this) return *this;
        rhs.info->replCopy(rhs.buff.data(), buff.data(), *info, N, N);
        info = rhs.info;
        return *this;
    }
    template <std::size_t N>
    template <std::size_t M>
    inline sized_any<N>& sized_any<N>::operator=(const sized_any<M>& rhs)
    {
        rhs.info->replCopy(rhs.buff.data(), buff.data(), *info, M, N);
        info = rhs.info;
        return *this;
    }
    template <std::size_t N>
    inline sized_any<N>& sized_any<N>::operator=(sized_any&& rhs) noexcept
    {
        if (&rhs == this) return *this;
        rhs.info->replMove(rhs.buff.data(), buff.data(), *info, N, N);
        info = rhs.info;
        rhs.info = &(sized_any_detail::info<void>);
        return *this;
    }
    template <std::size_t N>
    template <std::size_t M>
    inline sized_any<N>& sized_any<N>::operator=(sized_any<M>&& rhs) noexcept(M <= N)
    {
        if (&rhs == this) return *this;
        rhs.info->replMove(rhs.buff.data(), buff.data(), *info, M, N);
        info = rhs.info;
        rhs.info = &(sized_any_detail::info<void>);
        return *this;
    }
    template <std::size_t N>
    template <class ValueType>
    requires(std::conjunction_v<std::negation<sized_any_detail::is_sized_any<std::decay_t<ValueType>>>, std::is_copy_constructible<std::decay_t<ValueType>>>)
    inline sized_any<N>& sized_any<N>::operator=(ValueType&& rhs) noexcept(noexcept(sized_any{std::forward<ValueType>(rhs)}))
    {
        emplace<ValueType>(std::forward<ValueType>(rhs));
        return *this;
    }

    template <std::size_t N>
    template <class ValueType, class... Args>
    requires(std::copy_constructible<std::decay_t<ValueType>> && std::constructible_from<std::decay_t<ValueType>, Args...>)
    inline std::decay_t<ValueType> &sized_any<N>::emplace(Args&&... args) noexcept(noexcept(sized_any{in_place_type<ValueType>, std::forward<Args>(args)...}))
    {
        if constexpr (sized_any_detail::info<std::decay_t<ValueType>>.needsAlloc(N))
        {
            if (info->needsAlloc(N) && info->size() == sized_any_detail::info<std::decay_t<ValueType>>.size())
            {
                info->destroy(buff.data(), N);
                new (*reinterpret_cast<void**>(buff.data())) std::decay_t<ValueType>(std::forward<Args>(args)...);
            }
            else
            {
                info->del(buff.data(), N);
                *reinterpret_cast<void**>(buff.data()) = new std::decay_t<ValueType>(std::forward<Args>(args)...);
            }
            info = &(sized_any_detail::info<std::decay_t<ValueType>>);
            return **reinterpret_cast<std::decay_t<ValueType>**>(buff.data());
        }
        else
        {
            info->del(buff.data(), N);
            new (buff.data()) std::decay_t<ValueType>(std::forward<Args>(args)...);
            info = &(sized_any_detail::info<std::decay_t<ValueType>>);
            return *reinterpret_cast<std::decay_t<ValueType>*>(buff.data());
        }
    }
    template <std::size_t N>
    template<class ValueType, class U, class... Args>
    requires(std::copy_constructible<std::decay_t<ValueType>> && std::constructible_from<std::decay_t<ValueType>, std::initializer_list<U>&, Args...>)
    inline std::decay_t<ValueType>& sized_any<N>::emplace(std::initializer_list<U> il, Args&&... args) noexcept(noexcept(sized_any{in_place_type<ValueType>, il, std::forward<Args>(args)...}))
    {
        if constexpr (sized_any_detail::info<std::decay_t<ValueType>>.needsAlloc(N))
        {
            if (info->needsAlloc(N) && info->size() == sized_any_detail::info<std::decay_t<ValueType>>.size())
            {
                info->destroy(buff.data(), N);
                new (*reinterpret_cast<void**>(buff.data())) std::decay_t<ValueType>(il, std::forward<Args>(args)...);
            }
            else
            {
                info->del(buff.data(), N);
                *reinterpret_cast<void**>(buff.data()) = new std::decay_t<ValueType>(il, std::forward<Args>(args)...);
            }
            info = &(sized_any_detail::info<std::decay_t<ValueType>>);
            return **reinterpret_cast<std::decay_t<ValueType>**>(buff.data());
        }
        else
        {
            info->del(buff.data(), N);
            new (buff.data()) std::decay_t<ValueType>(il, std::forward<Args>(args)...);
            info = &(sized_any_detail::info<std::decay_t<ValueType>>);
            return *reinterpret_cast<std::decay_t<ValueType>*>(buff.data());
        }
    }

    template <std::size_t N>
    inline void sized_any<N>::reset() noexcept
    {
        info->del(buff.data(), N);
        info = &(sized_any_detail::info<void>);
    }
    template <std::size_t N>
    inline void sized_any<N>::swap(sized_any& other) noexcept
    {
        std::swap(buff, other.buff);
        std::swap(info, other.info);
    }

    template <std::size_t N>
    inline bool sized_any<N>::has_value() const noexcept
    {
        return info != &(sized_any_detail::info<void>);
    }
    template <std::size_t N>
    inline const std::type_info& sized_any<N>::type() const noexcept
    {
        return info->type();
    }

    template <class T, std::size_t N>
    T any_cast(const sized_any<N>& operand)
    {
        if (auto* casted = any_cast<std::remove_cvref_t<T>>(&operand)) return static_cast<T>(*casted);
        throw bad_any_cast{};
    }
    template <class T, std::size_t N>
    T any_cast(sized_any<N>& operand)
    {
        if (auto* casted = any_cast<std::remove_cvref_t<T>>(&operand)) return static_cast<T>(*casted);
        throw bad_any_cast{};
    }
    template <class T, std::size_t N>
    T any_cast(sized_any<N>&& operand)
    {
        if (auto* casted = any_cast<std::remove_cvref_t<T>>(&operand)) return static_cast<T>(std::move(*casted));
        throw bad_any_cast{};
    }
    template <class T, std::size_t N>
    const T* any_cast(const sized_any<N>* operand) noexcept
    {
        if (!operand || operand->info != &(sized_any_detail::info<std::decay_t<T>>)) return nullptr;
        if constexpr (sized_any_detail::info<std::decay_t<T>>.needsAlloc(N))
        {
            return *reinterpret_cast<const T**>(operand->buff.data());
        }
        else return reinterpret_cast<const T*>(operand->buff.data());
    }
    template <class T, std::size_t N>
    T* any_cast(sized_any<N>* operand) noexcept
    {
        if (!operand || operand->info != &(sized_any_detail::info<std::decay_t<T>>)) return nullptr;
        if constexpr (sized_any_detail::info<std::decay_t<T>>.needsAlloc(N))
        {
            return *reinterpret_cast<T**>(operand->buff.data());
        }
        else return reinterpret_cast<T*>(operand->buff.data());
    }

    template <std::size_t N, class T, class... Args>
    sized_any<N> make_sized_any(Args&&... args) noexcept(noexcept(sized_any<N>{in_place_type<T>, std::forward<Args>(args)...}))
    {
        return sized_any<N>{in_place_type<T>, std::forward<Args>(args)...};
    }
    template <std::size_t N, class T, class U, class... Args>
    sized_any<N> make_sized_any(std::initializer_list<U> il, Args&&... args) noexcept(noexcept(sized_any<N>{in_place_type<T>, il, std::forward<Args>(args)...}))
    {
        return sized_any<N>{in_place_type<T>, il, std::forward<Args>(args)...};
    }
    template<class T, class... Args>
    sized_any<sizeof(std::decay_t<T>)> make_sized_any(Args&&... args) noexcept(noexcept(make_sized_any<sizeof(std::decay_t<T>), T>(std::forward<Args>(args)...)))
    {
        return make_sized_any<sizeof(std::decay_t<T>), T>(std::forward<Args>(args)...);
    }
    template<class T, class U, class... Args>
    sized_any<sizeof(std::decay_t<T>)> make_sized_any(std::initializer_list<U> il, Args&&... args) noexcept(noexcept(make_sized_any<sizeof(std::decay_t<T>), T>(il, std::forward<Args>(args)...)))
    {
        return make_sized_any<sizeof(std::decay_t<T>), T>(il, std::forward<Args>(args)...);
    }
}