#define DTYPE float
extern DTYPE mode;
void DCNN(DTYPE image[513][513][3][1], 
  DTYPE p1out[257][257][16][1], 
  DTYPE p2out[129][129][16][1], 
  DTYPE p3out[65][65][32][1], 
  DTYPE p4out[33][33][16][1], 
  DTYPE p5out[17][17][16][1], 
  DTYPE p6out[9][9][16][1], 
  DTYPE p7out[5][5][3][1], 
  DTYPE output[3]);
#define IMG_H 512
#define IMG_W 512
#define IMG_C 3
#define SC_K 3
#define SC_S 2
#define SC_C 16
#define SC_H IMG_H/SC_S
#define SC_W IMG_W/SC_S
#define DC1_K 3
#define DC1_S 2
#define DC1_C 16
#define DC1_H SC_H/DC1_S
#define DC1_W SC_W/DC1_S
#define DC2_K 3
#define DC2_S 2
#define DC2_C 32
#define DC2_H DC1_H/DC2_S
#define DC2_W DC1_W/DC2_S
#define DC3_K 3
#define DC3_S 2
#define DC3_C 16
#define DC3_H DC2_H/DC3_S
#define DC3_W DC2_W/DC3_S
#define DC4_K 3
#define DC4_S 2
#define DC4_C 16
#define DC4_H DC3_H/DC4_S
#define DC4_W DC3_W/DC4_S
#define DC5_K 3
#define DC5_S 2
#define DC5_C 3
#define DC5_H DC4_H/DC5_S
#define DC5_W DC4_W/DC5_S
#define DC6_K 3
#define DC6_S 2
#define DC6_C 3
#define DC6_H DC5_H/DC6_S
#define DC6_W DC5_W/DC6_S
