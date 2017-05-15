/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef ORG_TENSORFLOW_JNI_JNI_UTILS_H_  // NOLINT
#define ORG_TENSORFLOW_JNI_JNI_UTILS_H_  // NOLINT

#include <jni.h>
#include <string>
#include <vector>

#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"

class AAssetManager;

bool PortableReadFileToProto(const std::string& file_name,
                             ::google::protobuf::MessageLite* proto)
    TF_MUST_USE_RESULT;

// Deserializes the contents of a file into memory.
void ReadFileToProtoOrDie(AAssetManager* const asset_manager,
                          const char* const filename,
                          google::protobuf::MessageLite* message);

std::string GetString(JNIEnv* env, jstring java_string);

tensorflow::int64 CurrentWallTimeUs();

#endif  // ORG_TENSORFLOW_JNI_JNI_UTILS_H_
