#include<stdio.h>
#include<stdlib.h>
#include <math.h>


int main(){
int **mat1, **mat2,**res,i,j,r1,c1,r2,c2,k;

printf("\n Order of the First matrix...\n");
scanf("%d %d",&r1,&c1);
printf("\nOrder of the Second matrix...\n");
scanf("%d %d",&r2,&c2);

if(c1!=r2){
    printf("wrong Order of matrix");
    exit(EXIT_SUCCESS);
}

mat1= (int**) malloc(r1*sizeof(int*));

for(i=0;i<c1;i++)
    mat1[i]=(int*)malloc(c1*sizeof(int));

mat2= (int**) malloc(r2*sizeof(int*));

for(i=0;i<c2;i++)
    mat2[i]=(int*)malloc(c2*sizeof(int));

res=(int**)calloc(r1,sizeof(int*));

for(i=0;i<c2;i++)
    res[i]=(int*)calloc(c2,sizeof(int));


    for(i=0;i<r1;i++)
        for(j=0;j<c1;j++)
            mat1[i][j] = ( int ) rand();

    for(i=0;i<r2;i++)
        for(j=0;j<c2;j++)
            mat2[i][j] = ( int ) rand();

printf("\n Entered Matrix 1: \n");
for(i=0;i<r1;i++){
    for(j=0;j<c1;j++)
        printf("%d ",mat1[i][j]);
    printf("\n");
}

printf("\n Entered Matrix 2: \n");
for(i=0;i<r2;i++){
    for(j=0;j<c2;j++)
        printf("%d ",mat2[i][j]);
    printf("\n");
}       



    for(i=0;i<r1;i++){
        for(j=0;j<c2;j++){
                res[i][j]=0;
                for(k=0;k<c1;k++)
                    res[i][j]+= mat1[i][k]*mat2[k][j];

        }
        printf("\n");
    }


   printf("\nThe Multiplication of two matrix is\n");
   for(i=0;i<r1;i++){
       printf("\n");
       for(j=0;j<c2;j++)
            printf("%d\t",res[i][j]);   
   }
    printf("\n");

return 0;}
