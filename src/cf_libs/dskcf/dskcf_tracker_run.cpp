#include "dskcf_tracker_run.hpp"
#include "dskcf_tracker.hpp"

DskcfTrackerRun::DskcfTrackerRun() : TrackerRun( "DSKCF" )
{
}

DskcfTrackerRun::~DskcfTrackerRun()
{
}

CfTracker * DskcfTrackerRun::parseTrackerParas(TCLAP::CmdLine& cmd, int argc, const char** argv)
{
	TCLAP::SwitchArg rawDepth( "", "raw_depth", "", cmd, false );
	TCLAP::SwitchArg rawColour( "", "raw_colour", "", cmd, false );
	TCLAP::SwitchArg rawConcatenate( "", "raw_concatenate", "", cmd, false );
	TCLAP::SwitchArg rawLinear( "", "raw_linear", "", cmd, false );
	TCLAP::SwitchArg hogColour( "", "hog_colour", "", cmd, false );
	TCLAP::SwitchArg hogDepth( "", "hog_depth", "", cmd, false );
	TCLAP::SwitchArg hogConcatenate( "", "hog_concatenate", "", cmd, true );
	TCLAP::SwitchArg hogLinear( "", "hog_linear", "", cmd, false );

	cmd.parse( argc, argv );

	return new DskcfTracker( );
}
