cmake_minimum_required(VERSION 2.8.3)
project(sm_permutohedral)

find_package(catkin REQUIRED cmake_modules)
include_directories(${catkin_INCLUDE_DIRS})
find_package(Boost REQUIRED COMPONENTS system)
find_package(Eigen REQUIRED)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++0x -D__STRICT_ANSI__")

include_directories(include ${Eigen_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})

catkin_package(
  INCLUDE_DIRS include ${catkin_INCLUDE_DIRS}
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS 
  DEPENDS
)

add_library(${PROJECT_NAME}
  src/permutohedral.cpp
)
target_link_libraries(${PROJECT_NAME} 
                      ${catkin_LIBRARIES} ${Boost_LIBRARIES})


##################
## Installation ##
##################

install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)


## Mark cpp header files for installation
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.hpp"
  PATTERN ".svn" EXCLUDE
)
