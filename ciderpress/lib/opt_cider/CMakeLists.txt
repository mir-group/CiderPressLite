
add_library(cider SHARED
  nr_cider_numint.c opt_cider_bas.c sph_harm.c spline.c
  efficient_partition.c
  gen_grid_partition.c nr_cider_numint_deprecated.c
  numint_util.c fractional_laplacian.c
)

set_target_properties(cider PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMPILE_FLAGS ${OpenMP_C_FLAGS}
  LINK_FLAGS ${OpenMP_C_FLAGS})

target_link_libraries(cider ${BLAS_LIBRARIES})
