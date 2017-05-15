/*
 * Author: Tomasz Grel (tomasz.grel@codilime.com)
 */


#ifndef DEBUG_UTILS_H
#define DEBUG_UTILS_H

#include <string.h>
#include <map>
#include <vector>
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <time.h>

#include "tensorflow/core/platform/logging.h"

struct Profiler {
  static inline long long int getWallClockNanos() {
    struct timespec tv;
    int errcode;

    errcode = clock_gettime(CLOCK_MONOTONIC,&tv);

    return 1000000000ll * tv.tv_sec + tv.tv_nsec;
  }

  static inline double getWallClockSeconds() {
    return getWallClockNanos() / 1e9;
  }

  double start;
  double previous_checkpoint;
  std::string name;

  inline Profiler(const std::string& name) {
    LOG(DEBUG) << "Measuring: " << name;
    start = getWallClockSeconds();
    previous_checkpoint = start;
    this->name = name;
  }

  inline ~Profiler() {
    double stop = getWallClockSeconds();
    LOG(DEBUG) << name << " (end) took: "
	      << stop - start;
  }

  inline void checkpoint(const std::string& message) {
    double currentTime = getWallClockSeconds();
    LOG(INFO) << name <<  " (" << message << ") time: \t"
              << currentTime - previous_checkpoint << " [s]";
    previous_checkpoint = currentTime;
  }
};

void inline print_array(float* data, size_t size, const std::string &prefix) {
	LOG(DEBUG) << "Array size: " << size;
	for (int i = 0; i != size; ++i) {
	  LOG(DEBUG) << prefix << ": " << data[i];
	}
}

void inline print_max_in_array(float* data, size_t size, const std::string &prefix) {
	auto max_it = std::max_element(data, data+size);
	LOG(DEBUG) << prefix << ", size: "<< size <<  "max: " << *max_it;
}

#endif
