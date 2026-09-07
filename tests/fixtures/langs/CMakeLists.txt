cmake_minimum_required(VERSION 3.10)

function(compute_total OUT)
  set(${OUT} 0 PARENT_SCOPE)
endfunction()

macro(render WIDTH)
  message(STATUS "${WIDTH}")
endmacro()
