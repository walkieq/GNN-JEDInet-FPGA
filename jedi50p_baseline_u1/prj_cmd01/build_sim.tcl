
open_project -reset jedi_prj

set_top jedi
add_files ../jedi.cpp -cflags "-std=c++0x"
add_files -tb ../tb_jedi.cpp -cflags "-std=c++0x"
open_solution -reset "solution1"

catch {config_array_partition -maximum_size 4096}
config_compile -name_max_length 60
set_part {xcu250-figd2104-2L-e}
create_clock -period 5 -name default


puts "***** C SIMULATION *****"
csim_design



exit
