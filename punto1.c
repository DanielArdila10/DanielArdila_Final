#include <omp.h>
#include <stdio.h>
#include <math.h>


double gaussian(double point);


double gauusian(double point)
{
	return exp(((-point*point)/2.0));
}

int main(int argc, char ** argv)
{
#pragma omp parallel{

	int thread = omp_get_thread_num();

	char arch[64];
	sprintf(arch, "cadena%d.dat",thread);
	FILE *file = fopen(arch, "wb");

	int n_points=100;
	int i;
	double point;
	double next_point;
	double alpha;
	double r;
	double sigma;
	float sigma;
	
	point=drand48();
	sigma=1.0
	
	for (i=0;i<n_points;i++)
	{
		next_point = point + sigma * (drand48()-0.5);    
			if(gaussian(next_point)/gaussian(point)>1.0){
					r=1.0;
				}
			else
			{
				r=gaussian(next_point)/gaussian(point);
			}  
		    	alpha = drand48();      
		    	if(alpha < r){
			      		point = next_point;
			    	}

			fprintf(file,"%lf\n", point);


		}	
		fclose(file);
	}


}
