// See LICENSE for license details.
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include "hbird_sdk_soc.h"
#include "oled.h"
#include "model.h"
#include "insn.h"
#include "yolo.h"
#include "nice.h"
#include "uart.h"
// A22 LED4    A23 LED5
// I2C0 UART2
// key3 4 A20 A21


void test()
{
	nice_ram_test();
	nice_fpu_test();
	ddr_test();
	nice_cont_test();
	led_test();
	oled_test();

	read_ddr((int *)0xA0001000,2);

	delay_1ms(100);
	eth_test();
	//DCNN_test();
}
int cam_lor=150, cam_uod=138;
int yolo_post(char flag, char cam_flag, char eth_flag)
{

	nice_cam_on();
	delay_1ms(50);
	nice_cam_off();
	nice_dcnn_on();
	delay_1ms(180);
	if(eth_flag)nice_eth_off();
	nice_hdmi_on();
	delay_1ms(10);nice_hdmi_off();
	float heatmap[HWOUT][HWOUT][COUT];
	memcpy((void*)heatmap,(void*)0xA0000000,HWOUT*HWOUT*COUT*4);
	float boxes[MAX_BOXNUM][NUM_CLASS + 5] = {0};
	yolo_head(heatmap, boxes);
	float nmsboxes[MAX_BOXNUM][NUM_CLASS + 5] = { 0 };
	int numofbox = NMS(boxes, nmsboxes);
	if(flag)boxprint(nmsboxes);
	hdmibox(nmsboxes, (0xA0B00000));
	if(eth_flag)nice_eth_on();
	nice_dcnn_off();
	if(cam_flag && (numofbox==1))
	{
		int x=(nmsboxes[0][0]+nmsboxes[0][2])/2;
		int y=(nmsboxes[0][1]+nmsboxes[0][3])/2;
		printf("%d, %d\n",x,y);
		if(y<85) cam_uod-=4;
		else if(y>170)cam_uod+=4;

		if(x<85) cam_lor+=5;
		else if(x>170)cam_lor-=5;

		nice_yuntai(cam_lor, cam_uod);
	}
	return numofbox;
}
void application()
{
	char zeros[50]={0};
	char buf[50]={0};
	char flag=10;
	char en_run_cam=0;
	char en_run_eth=0;
	OLED_ShowStr(2, 5, "cam run off",11, 1);
	OLED_ShowStr(1, 1, "ID:",3, 1);
	nice_eth_udp_len(120,480);

	//nice_eth_off();
	while(1){
		if((UART2->LSR&0x1)==1)//串口收到二维码
		{
			//printf("\n!!\n");
			uart_user_read(UART2, buf, 0);
			printf("ID:%s\n",buf);
			//for (int i=0;i<strlen(buf);i++) buf[i]=buf[i]+'0';
			OLED_ShowStr(19, 1, buf, 10, 1);memcpy(buf,zeros,50);
			printf("\n----------------------------\n");

		}
		if(gpio_read(GPIOA, KEY3)==0)
		{
			en_run_cam=!en_run_cam;
			if(en_run_cam)OLED_ShowStr(2, 5, "cam run on ",11, 1);
			else OLED_ShowStr(2, 5, "cam run off",11, 1);
		}
		if(gpio_read(GPIOA, KEY4)==0)
		{
			en_run_eth=!en_run_eth;
			if(en_run_eth)OLED_ShowStr(2, 6, "eth run on ",11, 1);
			else OLED_ShowStr(2, 6, "eth run off",11, 1);
			nice_eth_off();
		}
		int nump = yolo_post(flag, en_run_cam, en_run_eth);
	}
}
int main(void)
{
	gpio_config();
	OLED_Init();
	OLED_ON();
	OLED_CLS();
	uart_user_init(UART2);//初始化串口接收任意长度字符串
	uart_enable_tx_empt_int(UART2);
	ddr_init((int *)0xA1200000,0x00B00000,0);
	nice_cam_on();
	nice_hdmi_on(); delay_1ms(100);nice_hdmi_off();
	nice_yuntai(cam_lor, cam_uod);
	while(1)
		application();

	printf("\n/*********end**********/\n");
	return 0;
}





