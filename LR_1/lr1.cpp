#include <stdio.h>
#include <stdlib.h>
#include <chrono>

void subtraction(double * a, double * b, double * result, long long n){

    int i;
    for(i=0; i<n; ++i){
        result[i] = a[i] - b[i];
    }

}


int main(){

    long long n,i;
    scanf("%llu", &n);

    double* a = (double*)malloc(n * sizeof(double));
    double* b = (double*)malloc(n * sizeof(double));
    double* result = (double*)malloc(n * sizeof(double));

    for(i=0; i<n; ++i){
        scanf("%lf",&a[i]);
    }

    for(i=0; i<n; ++i){
        scanf("%lf",&b[i]);
    }
    std::chrono::steady_clock::time_point start = 
        std::chrono::steady_clock::now();

    subtraction(a,b,result,n);

    std::chrono::steady_clock::time_point finish = 
        std::chrono::steady_clock::now();
    unsigned time = 
        std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count(); 

    for(i=0; i<n; ++i){
        printf("%.10lf ", result[i]);
    }
    printf("\ntime: %dns\n", time);

    free(a);
    free(b);
    free(result);



    return 0;
}