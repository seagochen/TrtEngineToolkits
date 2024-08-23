# Compile the protobuf files to python files
protoc --python_out=./pys video_frame.proto
protoc --python_out=./pys inference_result.proto

# Compile the protobuf files to C++ files
protoc --cpp_out=./cpp video_frame.proto
protoc --cpp_out=./cpp inference_result.proto