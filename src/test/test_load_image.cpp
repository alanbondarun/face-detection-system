#include "utils/load_image.hpp"
#include <stdio.h>

using namespace NeuralNet;

int main()
{
	int i,j;
    FILE* fp=fopen("C:\\Users\\junse\\Downloads\\colortemplate.bmp","r");

    if (!fp)
    {
        printf("error loading file at fopen()\n");
        return 1;
    }

    unsigned char info[54];
    fread(info, sizeof(unsigned char), 54, fp);

    Image image;
    image.width = *(int*)(&info[18]);
    image.height =  *(int*)(&info[22]);

    printf("w=%d, h=%d\n", image.width, image.height);

    int row_padded = (image.width*3 + 3)&(~3);
    image.value = new double[row_padded];
    double tmp;

    unsigned char *gg = new unsigned char[row_padded];

    for(i = 0; i<(int)image.height; i++)
    {
        fread(gg, sizeof(unsigned char), row_padded, fp);
        printf("rgb = %d, %d, %d\n", gg[0], gg[1], gg[2]);
        /*for(j = 0; j<(int)image.width*3; j += 3)
        {
            tmp = image.value[j];
            image.value[j] = image.value[j+2];
            image.value[j+2] = tmp;
            printf("R:%f G:%f B:%f\n",image.value[j],image.value[j+1],image.value[j+2]);
        }*/
    }

    /*int row_padded = (image.width*3 + 3) & (~3);
    double* data = new double[row_padded];

    image.value = new double*[3];
    for(j=0;j<3; j++){
        image.value[j] = new double[image.width*image.height];
    }

    for(i = 0; i<(int)image.height; i++)
    {
        fread(data, sizeof(double), row_padded, fp);
        for(j = 0; j<(int)image.width*3; j += 3)
        {
            image.value[0][i*image.width+j]= data[j+2];
            image.value[1][i*image.width+j] = data[j+1];
            image.value[2][i*image.width+j] = data[j];
            printf("R:%d G:%d B:%d\n",(int)data[j+2],(int)data[j+1],(int)data[j]);
        }
    }*/

    //delete [] value;
    fclose(fp);
    delete [] gg;
    delete [] image.value;
	return 0;
}
