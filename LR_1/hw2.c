#include <stdio.h>
#include <stdlib.h>


void bubble_sort(float* arr, int n){

    float tmp;
    int i,j;
    for(i=0; i<n-1; ++i){
        for(j=0; j<n-i-1; ++j){
            if(arr[j] > arr[j+1]){
                tmp = arr[j];
                arr[j] = arr[j+1];
                arr[j+1] = tmp;
            }
        }
    }

}

int main(){

    int n,i;
    scanf("%d", &n);
    float *arr;

    arr = (float*) malloc(n*sizeof(float));

    for(i=0; i<n; ++i){
        scanf("%f", &arr[i]);
    }

    bubble_sort(arr,n);

    for(i=0; i<n; ++i){
        printf("%e ", arr[i]);
    }

    free(arr);

    return 0;
}
