#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MINVAL 1e-7

#define CSC(call) {                                                   \
	 cudaError err = call;                                             \
	 if(err!=cudaSuccess)                                              \
	 {                                                                  \
		 fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",   \
            __FILE__, __LINE__, cudaGetErrorString(err));				 \
	 }                                                                    \
 } while (0)

   
__global__ void Permute(double* Dev_Mtr, int* i, int* k, int* Dev_size)
{
	int index=blockDim.x*blockIdx.x+threadIdx.x;

	if(index<*Dev_size)
	{
		double tmp=Dev_Mtr[index*(*Dev_size)+(*i)];
		Dev_Mtr[index*(*Dev_size)+(*i)]=Dev_Mtr[index*(*Dev_size)+(*k)];
		Dev_Mtr[index*(*Dev_size)+(*k)]=tmp;	
	}

}


__global__ void MaxElement(double* Mtr, int Size, int i, int*strnum)
{
	double MaxValue=Mtr[i*Size+i];
	*strnum=i;

	for(int k=i; k<Size; k++)
	  {

		  if(fabs(Mtr[i*(Size)+k])>fabs(MaxValue))
			  {
				  *strnum=*strnum+1;    //это для компилятора чекера
				  *strnum=k;
				  MaxValue=Mtr[i*(Size)+k]; 
		      }
	  }

	if(fabs(MaxValue)<MINVAL)   //если максимальный элемент ниже порогового значения, то возвращаем -1 -> определитель равен 0 и выходим из цикла
	{
	  *strnum=-1;
	}

}

__global__ void Gaus(double* Mtr, int Size, int i)
{
	int index=blockIdx.x*blockDim.x+threadIdx.x;
	if(index>i && index< Size)
	{
		double particial = -Mtr[i*Size+index]/Mtr[i*Size+i];

        for(int z=i; z<Size; z++)                             
			{
				Mtr[z*Size+index]=Mtr[z*Size+index]+Mtr[z*Size+i]*particial;
			}		

	}
}

int main()
{
	int Size;
	int hostDet=1;    
	int HSTcountPerm=0;     //счетчик перестановок на хосте
	scanf("%d", &Size);

	if (Size==0) return 0;
    
	double *Mtr = (double*)malloc(Size*Size*sizeof(double));

	for(int i=0; i<Size; i++)
	{
		for(int j=0; j<Size; j++)
			scanf("%lf", &Mtr[j*Size+i]);
	}

	int* dev_Size;   //размер матрицы, который передаем на девайс
    double* dev_Mtr; // сама матрица которую передаем на девайс
	CSC(cudaMalloc((void**)&dev_Size, sizeof(int)));
	CSC(cudaMalloc((void**)&dev_Mtr, Size*Size*sizeof(double)));
	

	CSC(cudaMemcpy(dev_Size, &Size, sizeof(int), cudaMemcpyHostToDevice));
	CSC(cudaMemcpy(dev_Mtr, Mtr, Size*Size*sizeof(double), cudaMemcpyHostToDevice));

	int * Curr_str=NULL;
	int* New_Main_str=NULL;

	CSC(cudaMalloc((void**)&Curr_str, sizeof(int)));
	CSC(cudaMalloc((void**)&New_Main_str, sizeof(int)));
	
	int HostNewMainString=0;
	for(int i=0; i<Size; i++)
	{
		
		cudaMemcpy(New_Main_str, &i, sizeof(int), cudaMemcpyHostToDevice);
		MaxElement<<<1,1>>>(dev_Mtr, Size, i, New_Main_str);
		cudaMemcpy(&HostNewMainString, New_Main_str, sizeof(int), cudaMemcpyDeviceToHost);
		  
		if (HostNewMainString==-1) { hostDet=0; break;} 
		
			if(HostNewMainString!=i)
		{
			CSC(cudaMemcpy(Curr_str, &i, sizeof(int), cudaMemcpyHostToDevice));
			//CSC(cudaMemcpy(dev_Mtr, Mtr, Size*Size*sizeof(double), cudaMemcpyHostToDevice));
			Permute<<<100,100>>>(dev_Mtr, Curr_str, New_Main_str, dev_Size);
		    //CSC(cudaMemcpy(Mtr, dev_Mtr, Size*Size*sizeof(double), cudaMemcpyDeviceToHost));
			HSTcountPerm++;
		
		}

		Gaus<<<100,100>>>(dev_Mtr, Size, i);
	}

	double *ResMtr = (double*)malloc(Size*Size*sizeof(double));

	CSC(cudaMemcpy(ResMtr, dev_Mtr, Size*Size*sizeof(double), cudaMemcpyDeviceToHost));

	if(hostDet==0)
	{
		double ans=0;
		printf("%e ", ans);
		//system("pause");
		return 0;
	}
      
	double Det;
	int CountNegativeElements=0;

	if(ResMtr[0]<0) CountNegativeElements++;
	Det=log(fabs(ResMtr[0]));

	for(int i=1; i<Size; i++)
	{
		if(ResMtr[i*Size+i]<0) CountNegativeElements++;
		Det+=log(fabs(ResMtr[i*Size+i]));
	}

	Det=pow(exp(1.0), Det)*pow(-1.0, CountNegativeElements);

	if((HSTcountPerm % 2)!=0) Det*=-1;
	   
	printf("%e", Det);
	//getchar();	

    return 0;
}