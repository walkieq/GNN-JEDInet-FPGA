# GNN-JEDInet-FPGA

This repository includes a HLS-based template for the GNN-based JEDI-net with many hardware optimizaitons. These example designs are tested using Vivado HLS 2019.02. We are still work on more examples which will be released later. If you find any issue, please ping me an email.


## Example1: jedi50p_opt_acc
This is the design with the optimal accuracy and a low latency less than 1us.  

- How to run: 
```batch
cd jedi50p_opt_acc/prj_cmd01
vivado_hls -f build.tcl
```

- How to run only C-simulation: 
```batch
cd jedi50p_opt_acc/prj_cmd01
vivado_hls -f build_sim.tcl
```


The reports are in the following directory: prj_cmd01/jedi_prj/solution1/syn/report/

## Other Examples:
Same to the steps above using a different directory. 


## Notes:
jedi50p_opt_acc and jedi50p_opt_latn take us less than 1 hour to finish using a server with a Gold 6154 CPU. 
But jedi50p_baseline_u1 takes us 7 hours to finish the c-synthesis on the same server. 


