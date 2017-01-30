#include "dskcf_tracker_run.hpp"

int main( int argc, const char** argv )
{
  DskcfTrackerRun main;

  if( main.start( argc, argv ) )
  {
    return 0;
  }
  else
  {
    return -1;
  }
}
