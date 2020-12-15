#!/bin/bash -l

sacct -u dc-alta2 --format=JobID,Jobname%40,partition,state,time,Submit,start,end,elapsed,MaxRss,MaxVMSize,nnodes,ncpus,nodelist,CPUTime,AveRSS,MaxRSS,ExitCode