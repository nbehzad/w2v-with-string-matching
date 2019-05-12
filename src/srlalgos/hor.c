#include "hor.h"
#include "include/define.h"

#include "stdio.h"
#include "stdlib.h"
#include "string.h"

static void pre_horspool(char *pattern, int pattern_size, unsigned char hbc[]) {
  int i;
  for (i=0;i<SIGMA;i++)   hbc[i]=pattern_size;
  for (i=0;i<pattern_size-1;i++) hbc[pattern[i]]=pattern_size - i - 1;
}

void hor(search_parameters params) {
  int i, s;
  unsigned char hbc[SIGMA];
  
  pre_horspool(params.pattern, params.pattern_size, hbc);

  /* Searching */
  s = 0;
  while(s <= params.text_size - params.pattern_size) {
    i=0;
    while(i < params.pattern_size && params.pattern[i] == params.text[s + i]) i++;
    if (i == params.pattern_size) params.match[0]++;
    s += hbc[params.text[s + params.pattern_size - 1]];
  }
}
