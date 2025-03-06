
add_lab("TiledMatrixMultiplication")
add_lab_solution("TiledMatrixMultiplication" ${CMAKE_CURRENT_LIST_DIR}/no_timer_util.cu)
add_generator("TiledMatrixMultiplication" ${CMAKE_CURRENT_LIST_DIR}/dataset_generator.cpp)
