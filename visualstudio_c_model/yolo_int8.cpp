
#include <iostream>
#include "yolo_weights.h"
#include "defines.h"
#include "pic_fpga.h"
#include "cmath"
#include "windows.h"
#include <time.h>
/*
 * Anchors
 */
static const float anchors[NUM_ANCHOR][2] = { {50,50}, {71,74}, {98,91} };


void img_pre_process(uint8 in[IMGHW][IMGHW][3], DTYPE out[IMGHW+2][IMGHW+2][3])
{
    for (int16 i = 0; i < IMGHW; i++)
        for (int16 j = 0; j < IMGHW; j++)
            for (int16 k = 0; k < 3; k++)
            {
                out[i + 1][j + 1][k] = in[i][j][k] /255.0;
            }
}
/*
 * sigmoid function
 */
float sigmoid(float x)
{
	float y = 1 / (1 + expf(-x));
	return y;
}

/*
 * Sort boxes according to confidence
 * boxes: boxes store sequence
 * box : new boxes to insert
 */
int sort(float boxes[MAX_BOXNUM][NUM_CLASS + 5], float box[NUM_CLASS + 5])
{
	for (uint16_t i = 0; i < MAX_BOXNUM; i++)
	{
		if (box[4] > boxes[i][4])
		{
			for (uint16_t j = MAX_BOXNUM - 1; j > i; j--)
			{
				memcpy(boxes[j], boxes[j - 1], (NUM_CLASS + 5) * 4);
			}
			memcpy(boxes[i], box, (NUM_CLASS + 5) * 4);
			return 0;
		}
	}
	return 0;
}
/*
 * Attention!!! Please initialize @para nmsboxes with zeros
 */
int NMS(float boxes[MAX_BOXNUM][NUM_CLASS + 5], float nmsboxes[MAX_BOXNUM][NUM_CLASS + 5])
{
	uint16_t valid_box;
	float S[MAX_BOXNUM];
	//查看有几个有效boxes�? !!!请将无效boxes 中conf 归零
	if (boxes[0][4] > CONF_THRESHOLD)memcpy(nmsboxes[0], boxes[0], (NUM_CLASS + 5) * 4);
	else return 0;
	for (uint16_t i = 0; i < MAX_BOXNUM; i++)
	{
		if (boxes[i][4] < CONF_THRESHOLD)
		{
			valid_box = i;
			break;
		}
		else S[i] = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]);
	}
	uint16_t index_nms_boxes = 1;

	for (uint16_t i = 1; i < valid_box; i++)
	{
		float maxiou = 0;
		for (uint16_t j = 0; j < i; j++)
		{
			float Si = (MIN(boxes[j][2], boxes[i][2]) - MAX(boxes[j][0], boxes[i][0])) * (MIN(boxes[j][3], boxes[i][3]) - MAX(boxes[j][1], boxes[i][1]));
			float iou = Si / (S[j] + S[i] - Si);
			if (iou > maxiou) maxiou = iou;
		}
		if (maxiou < NMS_IOU)
		{
			memcpy(nmsboxes[index_nms_boxes], boxes[i], (NUM_CLASS + 5) * 4);
			index_nms_boxes++;
		}
	}
	return 0;
}

/*
 * yolo_head
 * Boxes are the output of whole model.
 * void * data: output of ai model
 * boxes : provide MAX_BOXNUM boxes (x1,y1, x2,y2, confidence, mark0, mark1)
 */
int yolo_head(float out_data[HWOUT][HWOUT][COUT], float boxes[MAX_BOXNUM][NUM_CLASS + 5])
{
	int grid_x, grid_y;
	float x, y, w, h;
	float x1, y1, x2, y2;
	float box[NUM_CLASS + 5];
	for (int i = 0; i < FEATHW; i++)
	{
		for (int j = 0; j < FEATHW; j++)
		{
			for (int k = 0; k < NUM_ANCHOR; k++)
			{
				// 其中21维度包含了每个像素预测的三个锚框，每个锚框对7个维度，依次为x y w h conf1 conf2 class
				// 当然因为这个网络是单类检测，以class这一维度没有
				float conf = sigmoid(out_data[i][j][k * (NUM_CLASS + 5) + 4]);
				float mark0 = sigmoid(out_data[i][j][k * (NUM_CLASS + 5) + 5]);
				float mark1 = sigmoid(out_data[i][j][k * (NUM_CLASS + 5) + 6]);

				//置信程度返回坐标
				if (conf > CONF_THRESHOLD)
				{
					x = out_data[i][j][k * (NUM_CLASS + 5) + 0];
					y = out_data[i][j][k * (NUM_CLASS + 5) + 1];
					w = out_data[i][j][k * (NUM_CLASS + 5) + 2];
					h = out_data[i][j][k * (NUM_CLASS + 5) + 3];
					x = (sigmoid(x) + j + 0) * ZOOM;
					y = (sigmoid(y) + i + 0) * ZOOM;
					//w = expf(w) * anchors[k][0];
					//h = expf(h) * anchors[k][1];
					w = 2*sigmoid(w) * anchors[k][0];
					h = 2*sigmoid(h) * anchors[k][1];
					x1 = (x - w / 2);
					x2 = (x + w / 2);
					y1 = y - h / 2;
					y2 = y + h / 2;
					if (x1 < 0) x1 = 0;
					if (y1 < 0) y1 = 0;
					if (x2 > IMGHW - 1) x2 = IMGHW - 1;
					if (y2 > IMGHW - 1) y2 = IMGHW - 1;
					//左上角坐标为(x1, y1)，左下角坐标(x2, y2)
					box[0] = x1; box[1] = y1; box[2] = x2; box[3] = y2;
					box[4] = conf; box[5] = mark0; box[6] = mark1;
					sort(boxes, box);
				}
			}
		}
	}
	return 0;
}

/*
* boxes print 
*/
void boxprint(float nmsboxes[MAX_BOXNUM][NUM_CLASS + 5])
{
	printf("\n----------------------------\n");
	for(int16 i=0;i< MAX_BOXNUM;i++)
		if (nmsboxes[i][4] > CONF_THRESHOLD)
		{
			if (nmsboxes[i][5] > nmsboxes[i][6])
				printf("-masked");
			else 
				printf("-nomask");
			printf("-box: conf:%.3f (%.2f,%.2f) (%.2f,%.2f)\n",
				nmsboxes[i][4], nmsboxes[i][0], nmsboxes[i][1], nmsboxes[i][2], nmsboxes[i][3]);
		}
	printf("\n----------------------------\n");
}


int main()
{
	int i = 0;
	int begintime, endtime;
	LARGE_INTEGER _start = { 0 }, _end = { 0 };
	static DTYPE yout1[HW1 + 2][HW1 + 2][C1] = { 0 };
	static DTYPE yout2[HW1 + 2][HW1 + 2][C1] = { 0 };
	static DTYPE yout3[HW2 + 2][HW2 + 2][C2] = { 0 };
	static DTYPE yout4[HW2 + 2][HW2 + 2][C2] = { 0 };

	static DTYPE yout5[HW3 + 2][HW3 + 2][C3] = { 0 };
	static DTYPE yout6[HW3 + 2][HW3 + 2][C3] = { 0 };
	static DTYPE yout7[HW4 + 2][HW4 + 2][C4] = { 0 };
	static DTYPE yout8[HW4 + 2][HW4 + 2][C4] = { 0 };
	static DTYPE yout9[HW5 + 2][HW5 + 2][C5] = { 0 };
	static DTYPE yout10[HW5 + 2][HW5 + 2][C5] = { 0 };

	std::cout << "\nTEMPLATE - initialization\n";
	static DTYPE img[OH][OW][3] = {0};
	static DTYPE imgin[IMGHW + 2][IMGHW + 2][3] = {0};
	static float modelout[HWOUT][HWOUT][COUT] = { 0 };
    //img_pre_process(m1, imgin);
   //yolo_net_test(imgin, yout1, yout2, yout3, yout4, yout5,
    //		yout6, yout7, yout8, yout9, yout10, modelout);
	
	begintime = clock();
	__asm {
		rdtsc
		mov dword ptr[_start.LowPart], eax
		mov dword ptr[_start.HighPart], edx
	}
	while (i++ < 10){
		float nmsboxes[MAX_BOXNUM][NUM_CLASS + 5] = { 0 };
		unsigned short HDMIF[OH][OH] = { 0 };
		float boxes[MAX_BOXNUM][NUM_CLASS + 5] = { 0 };
		yolo_net(m1, HDMIF, img, imgin, yout1, yout2, yout3, yout4, yout5,
				yout6, yout7, yout8, yout9, yout10, modelout);
		yolo_head(modelout, boxes);
		NMS(boxes, nmsboxes);
		if(i==1)boxprint(nmsboxes);
	}
	__asm {
		rdtsc
		mov dword ptr[_end.LowPart], eax
		mov dword ptr[_end.HighPart], edx
	}
	endtime = clock();
	
	long interval = (_end.QuadPart - _start.QuadPart);
	printf("\n\nRunning Time (10 times)：%dms\n", endtime - begintime);
    std::cout <<"\nTEMPLATE - done once\n";

    return 0;
}

