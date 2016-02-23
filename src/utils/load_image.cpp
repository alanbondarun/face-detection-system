#include "utils/load_image.hpp"
#include <stdio.h>

namespace NeuralNet
{
	Image loadImage(const char* filepath)
	{
/*	    int i,j;
	    FILE* fp=fopen(filepath,"r");
	    unsigned char info[54];
	    fread(info, sizeof(unsigned char), 54, fp);

        Image image;
        image.width = *(int*)(&info[18]);
        image.height =  *(int*)(&info[22]);

        int psize = 3*image.width*image.height;
        double *data;
        data = new double[image.width*image.height];
        fread(data, sizeof(double), psize, fp);

        /* TODO */

/*        image.value = new double*[3];
        for(j=0;j<3; j++){
            image.value[j] = new double[image.width*image.height];
        }
		for(i=0;i<psize;i+=3){
            image.value[0][i/3]=data[i+2];
            image.value[1][i/3]=data[i+1];
            image.value[2][i/3]=data[i];
		}

        fclose(fp);
        delete [] data;*/

		Image i1;
		return i1;
	}
}
