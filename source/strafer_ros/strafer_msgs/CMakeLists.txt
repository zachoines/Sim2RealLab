cmake_minimum_required(VERSION 3.8)
project(strafer_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

# Custom messages -- add .msg files here as needed.
# Currently a placeholder. Uncomment and add messages when standard messages
# prove insufficient.
#
# rosidl_generate_interfaces(${PROJECT_NAME}
#   "msg/WheelVelocities.msg"
#   "msg/EncoderTicks.msg"
#   "msg/PolicyAction.msg"
#   DEPENDENCIES std_msgs
# )

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
