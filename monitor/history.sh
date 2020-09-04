#!/bin/bash -l

sacct -u dc-alta2 --format=JobID,Jobname%40,partition,state,time,start,end,elapsed,MaxRss,MaxVMSize,nnodes,ncpus,nodelist,CPUTime,AveRSS,MaxRSS,ExitCode