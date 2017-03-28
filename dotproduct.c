/*
 * blockAndthread.c
 * A "driver" program that calls a routine (i.e. a kernel)
 * that executes on the GPU.  The kernel fills two int arrays
 * with the block ID and the thread ID
 *
 * Note: the kernel code is found in the file 'blockAndThread.cu'
 * compile both driver code and kernel code with nvcc, as in:
 * 			nvcc blockAndThread.c blockAndThread.cu
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

//#define SIZEOFARRAY 10240000

struct timeval  ts1, ts2, tps1, tps2;
// The serial function dotproduct
void serial_dotproduct(long long *force, long long *distance, long long size, long long *result)
{
    int i;
    *result = 0;
    for (i = 0; i < size; i++)
        *result += force[i]*distance[i];
}

// The function dotproduct is in the file dotproduct.cu
extern void cuda_dotproduct(long long *force, long long *distance, long long size, long long *result_array, double *time_result);

int main (int argc, char *argv[])
{
    long long SIZEOFARRAY;
    if (argc !=  2){
    	printf("Usage: dotproduct <array_size>\n");
	exit(2);
    }else{
        SIZEOFARRAY = atoll(argv[1]);
    }
    //timeval tv1, tv2;
    // Declare arrays and initialize to 0
    long long *force;
    force = (long long*)malloc(SIZEOFARRAY*sizeof(long long));
    long long *distance;
    distance = (long long*)malloc(SIZEOFARRAY*sizeof(long long));
    long long *result_array;
    result_array = (long long*)malloc(SIZEOFARRAY*sizeof(long long));
    gettimeofday(&tps1, NULL);
    // Here's where I could setup the arrays.
    long long i;
    long long j = 0;
    long long k = 0;
    for (i=0; i < SIZEOFARRAY; i++) {
        if(i < SIZEOFARRAY/2){
            force[i]=i+1;
	}else{
	    force[i]=SIZEOFARRAY/2-j;
	    j++;
	}
	if (i%10 != 0){
	    distance[i]=k+1;
            k++;
	}else{
	    distance[i]=1;
	    k = 1;
	}
    }
    for(i=0;i<SIZEOFARRAY;i++){
        result_array[i]=0;
    }
    gettimeofday(&tps2, NULL);
    double tps_time = (double) (tps2.tv_usec - tps1.tv_usec) / 1000000 + (double) (tps2.tv_sec - tps1.tv_sec);
    
    // Serial dotproduct
    long long serial_result = 0;
    gettimeofday(&ts1, NULL);
    serial_dotproduct(force, distance, SIZEOFARRAY, &serial_result);
    gettimeofday(&ts2, NULL);
    double ts_time = (double) (ts2.tv_usec - ts1.tv_usec) / 1000000 + (double) (ts2.tv_sec - ts1.tv_sec);
    
    double cuda_time_result = 0.0; 
    // Call the function that will call the GPU function
    cuda_dotproduct (force, distance, SIZEOFARRAY, result_array, &cuda_time_result);
    
    long long cuda_result = 0;
    for (i = 0; i < SIZEOFARRAY; i++){
        cuda_result += result_array[i];
    }
    
    if(serial_result == cuda_result){
    	printf("array_size,ts,tps,tp\n");
	printf("%lld,%.10f,%.10f,%.10f\n",SIZEOFARRAY,ts_time,tps_time,cuda_time_result);
    }else{
        printf("ERROR in CUDA calculation");
    }
    return 0;
}

