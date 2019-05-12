
#ifndef HOR_CUH
#define HOR_CUH

#define SIGMA 256

__global__
void horspool(char *text, unsigned long text_size, char *pattern, int pattern_size,
     		         unsigned char hbc[], int stride_length, int *match);

void pre_horspool(char *pattern, int pattern_size, unsigned char hbc[]);

#endif