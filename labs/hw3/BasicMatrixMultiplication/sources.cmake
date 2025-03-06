
add_lab("BasicMatrixMultiplication")
add_lab_solution("BasicMatrixMultiplication" ${CMAKE_CURRENT_LIST_DIR}/solution_template_with_timer_utility_without_wb_library.cu)
add_generator("BasicMatrixMultiplication" ${CMAKE_CURRENT_LIST_DIR}/dataset_generator.cpp)
