#ifndef _OPTIONAL_HPP_
#define _OPTIONAL_HPP_

#include <exception>
#include <memory>

template< class T >
class optional
{
public:
  class bad_optional_access : public std::exception
  {
    virtual const char * what() const throw()
    {
      return "Optional value not set";
    }
  };

  optional()
  {
  }

  optional( const optional< T > & value )
  {
    this->m_value = std::make_shared< T >( value.m_value );
  }

  optional( optional< T > && value )
  {
    this->m_value = std::move( value.m_value );
  }

  optional( const T & value )
  {
    this->m_value = std::make_shared< T >( value );
  }

  optional( T && value )
  {
    this->m_value = std::make_shared< T >( std::move( value ) );
  }

  ~optional()
  {
  }

  optional< T > & operator=( const optional< T > & value )
  {
    this->m_value = std::make_shared< T >( value->value );

    return *this;
  }

  optional< T > & operator=( optional< T > && value )
  {
    this->m_value = std::move( value->m_value );

    return *this;
  }

  optional< T > & operator=( const T & value )
  {
    this->m_value = std::make_shared< T >( value );

    return *this;
  }

  optional< T > & operator=( T && value )
  {
    this->m_value = std::make_shared< T >( std::move( value->m_value ) );

    return *this;
  }

  const T * operator->() const
  {
    return this->m_value.get();
  }

  T * operator->()
  {
    return this->m_value.get();
  }

  const T & operator*() const
  {
    return *this->m_value;
  }

  T & operator*()
  {
    return *this->m_value;
  }

  operator bool() const
  {
    return ( this->m_value != nullptr );
  }

  T & value()
  {
    if( this->m_value )
    {
      return *this->m_value;
    }
    else
    {
      throw bad_optional_access();
    }
  }

  const T & value() const
  {
    if( this->m_value )
    {
      return *this->m_value;
    }
    else
    {
      throw bad_optional_access();
    }
  }
private:
  std::shared_ptr< T > m_value;
};

#endif
