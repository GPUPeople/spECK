#pragma once

#include <string>
#include <map>
#include "INIReader.h"

class Config
{
public:
  enum Key
  {
	  BlockNnzFillRatio,
	  MaxRowsPerBlock,
	  LoadBalanceScanMode,
	  HashScanSupportRestarts,
	  InputFile,
	  LoadBalanceModeCounting,
	  LoadBalanceModeNumeric,
	  ReprocessLoadBalancingForNumeric,
	  MaxNnzPerBlockNumeric,
	  MaxRowsPerBlockNumeric,
	  IterationsWarmUp,
	  IterationsExecution,
	  SupportGlobalFallback,
	  TrackIndividualTimes,
	  TrackCompleteTimes,
	  Debug,
	  SortMode,
	  CompareResult,
	  AutoSelectKernelSizeMode,
	  CountingBlockLimit,
	  LogsEnabled,
	  BlocksPerSM,
	  DenseThresholdNumericExternalSorting,
	  DenseThresholdNumericInternalSorting,
	  GlobalDenseThresholdNumeric,
	  DenseThresholdCounting,
	  SpGEMMMethodCounting,
	  SpGEMMMethodNumeric,
	  ThreadsPerNnzOffset,
	  add3MinLength,
	  add2MinLength,
	  add1MinLength,
	  add3MaxCols,
	  add2MaxCols,
	  add1MaxCols,
	  sub2MaxCols,
	  sub1MaxCols,
	  sub2MinThreads,
	  sub1MinThreads,
	  add2MinConcurrentOps,
	  add1MinConcurrentOps,
	  maxOpsWeight64,
	  maxOpsWeight128,
	  maxOpsWeight256,
	  maxOpsWeight512,
	  maxOpsWeight1024,
	  staticThreadsPerRow
  };

  enum ScanMode
  {
	  Std = 0,
	  Cub = 1,
	  Thrust = 2,
	  WorkEfficient = 3
  };

  enum SpGEMMMethods
  {
	  AutoSpGEMM = 0,
	  HashSpGEMM = 1,
	  DenseSpGEMM = 2
  };

  enum LoadBalanceModes
  {
	  AutoEnable = 0,
	  ForceEnable = 1,
	  ForceDisable = 2
  };

  enum SortModes
  {
	  None = 0,
	  Separate = 1,
	  InPlace = 2,
	  Auto = 3,
	  CubSegmentedSort = 4
  };

private:
	std::map<Key, std::string> keyToString;
	std::map<Key, int> overrides;
	INIReader reader;

	void addKeyToString() {
		keyToString = {
			{BlockNnzFillRatio, "BlockNnzFillRatio"},
			{MaxRowsPerBlock, "MaxRowsPerBlock"},
			{LoadBalanceScanMode, "LoadBalanceScanMode"},
			{HashScanSupportRestarts, "HashScanSupportRestarts"},
			{InputFile, "InputFile"},
			{ReprocessLoadBalancingForNumeric, "ReprocessLoadBalancingForNumeric"},
			{MaxNnzPerBlockNumeric, "MaxNnzPerBlockNumeric"},
			{MaxRowsPerBlockNumeric, "MaxRowsPerBlockNumeric"},
			{IterationsWarmUp, "IterationsWarmUp"},
			{IterationsExecution, "IterationsExecution"},
			{SupportGlobalFallback, "SupportGlobalFallback"},
			{TrackIndividualTimes, "TrackIndividualTimes"},
			{TrackCompleteTimes, "TrackCompleteTimes"},
			{Debug, "Debug"},
			{LoadBalanceModeCounting, "LoadBalanceModeCounting"},
			{LoadBalanceModeNumeric, "LoadBalanceModeNumeric"},
			{SortMode, "SortMode"},
			{CompareResult, "CompareResult"},
			{AutoSelectKernelSizeMode, "AutoSelectKernelSizeMode"},
			{CountingBlockLimit, "CountingBlockLimit"},
			{LogsEnabled, "LogsEnabled"},
			{BlocksPerSM, "BlocksPerSM"},
			{DenseThresholdNumericExternalSorting, "DenseThresholdNumericExternalSorting"},
			{DenseThresholdNumericInternalSorting, "DenseThresholdNumericInternalSorting"},
			{DenseThresholdCounting, "DenseThresholdCounting"},
			{SpGEMMMethodNumeric, "SpGEMMMethodNumeric"},
			{SpGEMMMethodCounting, "SpGEMMMethodCounting"},
			{GlobalDenseThresholdNumeric, "GlobalDenseThresholdNumeric"},
			{add3MinLength, "add3MinLength"},
			{add2MinLength, "add2MinLength"},
			{add1MinLength, "add1MinLength"},
			{add3MaxCols, "add3MaxCols"},
			{add2MaxCols, "add2MaxCols"},
			{add1MaxCols, "add1MaxCols"},
			{sub2MaxCols, "sub2MaxCols"},
			{sub1MaxCols, "sub1MaxCols"},
			{sub2MinThreads, "sub2MinThreads"},
			{sub1MinThreads, "sub1MinThreads"},
			{add2MinConcurrentOps, "add2MinConcurrentOps"},
			{add1MinConcurrentOps, "add1MinConcurrentOps"},
			{ThreadsPerNnzOffset, "ThreadsPerNnzOffset"},
			{maxOpsWeight64, "maxOpsWeight64"},
			{maxOpsWeight128, "maxOpsWeight128"},
			{maxOpsWeight256, "maxOpsWeight256"},
			{maxOpsWeight512, "maxOpsWeight512"},
			{maxOpsWeight1024, "maxOpsWeight1024"},
			{staticThreadsPerRow, "staticThreadsPerRow"} };
	}

	Config()
	{
		reader = INIReader();
		addKeyToString();
	}

	Config(std::string configPath)
	{
		reader = INIReader(configPath);
		addKeyToString();
	}

	static Config &Instance()
	{
		if (_instance == nullptr)
			throw std::exception();
	
		return *_instance;
	}

public:
	static Config *_instance;
	static void init(std::string path);
	static void init();

	static int getInt(Key key, int fallback = -1);
	static int setInt(Key key, int newVal);
	static std::string getString(Key key, std::string fallback = "");
	static bool getBool(Key key, bool fallback = false);
	static float getFloat(Key key, float fallback = 0.0);
};


