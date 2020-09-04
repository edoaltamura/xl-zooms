#!/bin/bash -l

sacct -u dc-alta2 --format=JobID,JobName%40,Partition,AveRSS,MaxRSS,AveVMSize,MaxVMSize,Elapsed,ExitCode