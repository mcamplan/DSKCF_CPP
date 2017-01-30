//
// Created by jake on 08/08/15.
//

#ifndef CFTRACKING_CIRCULARBUFFER_HPP
#define CFTRACKING_CIRCULARBUFFER_HPP

#include <sys/types.h>
#include <array>

template< class T, uint N >
class circularbuffer
{
public:
  circularbuffer()
  {
    this->index = 0;
  }

  void push_back( const T & value )
  {
    this->buffer[ this->index ] = value;

    this->index = ( this->index + 1 ) % N;
  }

  void push_back( T && value )
  {
    this->buffer[ this->index ] = std::move( value );

    this->index = ( this->index + 1 ) % N;
  }

  typename std::array< T, N >::const_iterator begin() const
  {
    return this->buffer.begin();
  }

  typename std::array< T, N >::const_iterator end() const
  {
    return this->buffer.end();
  };

  typename std::array< T, N >::iterator begin()
  {
    return this->buffer.begin();
  }

  typename std::array< T, N >::iterator end()
  {
    return this->buffer.end();
  };
private:
  uint index;
  std::array< T, N > buffer;
};

#endif //CFTRACKING_CIRCULARBUFFER_HPP
