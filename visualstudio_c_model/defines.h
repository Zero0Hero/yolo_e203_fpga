//#include "hls_half.h"
// type defines
#define DTYPE float

#define int32 int
#define int16 short
#define int8 char
#define uint8 unsigned char

// HW defines
#define OW 640
#define OH 480
#define IMGHW 256
#define HW1 (IMGHW/2)
#define C1 16
#define HW2 (HW1/2)
#define C2 16
#define HW3 (HW2/2)
#define C3 32
#define HW4 (HW3/2)
#define C4 32
#define HW5 (HW4/2)
#define C5 64

#define HWOUT HW5
#define COUT 21
#define ZOOM (IMGHW/HWOUT)
#define FEATHW HWOUT

// boxes defines
#define NUM_ANCHOR 3
#define NUM_CLASS 2 //������Ŀ 2
#define MAX_BOXNUM 10
#define NMS_IOU 0.1 //�Ǽ���ֵ���ƽ�������ֵ
#define CONF_THRESHOLD 0.15 //�滻Ϊ��������ų̶�???

// min max defines 
#define MAX(a,b) ((a>b)? a:b)
#define MIN(a,b) ((a<b)? a:b)

// functions defines 
void yolo_net_test(DTYPE in[IMGHW + 2][IMGHW + 2][3],
		DTYPE out1[HW1 + 2][HW1 + 2][C1], DTYPE out2[HW1 + 2][HW1 + 2][C1],
		DTYPE out3[HW2 + 2][HW2 + 2][C2], DTYPE out4[HW2 + 2][HW2 + 2][C2],
		DTYPE out5[HW3 + 2][HW3 + 2][C3], DTYPE out6[HW3 + 2][HW3 + 2][C3],
		DTYPE out7[HW4 + 2][HW4 + 2][C4], DTYPE out8[HW4 + 2][HW4 + 2][C4],
		DTYPE out9[HW5 + 2][HW5 + 2][C5], DTYPE out10[HW5 + 2][HW5 + 2][C5],
		float out[HWOUT][HWOUT][COUT]);

void yolo_net(unsigned short RGB565[OH][OW], unsigned short HDMIF[OH][OH],
		DTYPE IMG_P[OH][OW][3], DTYPE modelin[IMGHW + 2][IMGHW + 2][3],
		DTYPE out1[HW1 + 2][HW1 + 2][C1], DTYPE out2[HW1 + 2][HW1 + 2][C1],
		DTYPE out3[HW2 + 2][HW2 + 2][C2], DTYPE out4[HW2 + 2][HW2 + 2][C2],
		DTYPE out5[HW3 + 2][HW3 + 2][C3], DTYPE out6[HW3 + 2][HW3 + 2][C3],
		DTYPE out7[HW4 + 2][HW4 + 2][C4], DTYPE out8[HW4 + 2][HW4 + 2][C4],
		DTYPE out9[HW5 + 2][HW5 + 2][C5], DTYPE out10[HW5 + 2][HW5 + 2][C5],
		float out[HWOUT][HWOUT][COUT]);
